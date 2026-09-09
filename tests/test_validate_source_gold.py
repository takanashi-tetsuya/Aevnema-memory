from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import benchmarks.validate_source_gold as source_gold_validator
from benchmarks.validate_source_gold import (
    FROZEN_SOURCE_MANIFEST_SCHEMA,
    SOURCE_GOLD_VALIDATION_SCHEMA,
    SourceGoldValidationInputError,
    _artifact_input_rejection_code,
    _has_link_or_junction_component,
    _report_output_rejection_code,
    _source_root_rejection_code,
    _windows_locality_rejection_code,
    main,
    validate_source_gold,
    validate_source_gold_bytes,
    validate_source_gold_files,
)


_BODY_A = "SYNTHETIC_SOURCE_BODY_A_DO_NOT_REPORT"
_BODY_B = "SYNTHETIC_SOURCE_BODY_B_DO_NOT_REPORT"


def _sha256_bytes(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _codes(report: dict[str, object], field: str = "issues") -> set[str]:
    return {str(item["code"]) for item in report[field]}  # type: ignore[index]


class SourceGoldValidationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, object]:
        source_root = root / "sources"
        source_a = source_root / "main" / "a.json"
        source_b = source_root / "main" / "b.json"
        _write_json(source_a, {"content": [{"TextCn": _BODY_A}]})
        _write_json(source_b, {"content": [{"TextCn": _BODY_B}]})
        source_hashes = {
            "main/a.json": _sha256_bytes(source_a),
            "main/b.json": _sha256_bytes(source_b),
        }
        source_manifest = {
            "schema": FROZEN_SOURCE_MANIFEST_SCHEMA,
            "sources": [
                {
                    "source_key": source_key,
                    "source_file_sha256": source_hashes[source_key],
                }
                for source_key in sorted(source_hashes)
            ],
        }
        source_manifest_path = root / "frozen-source-manifest.json"
        _write_json(source_manifest_path, source_manifest)

        def atom(
            source_key: str,
            body: str,
            *,
            structured_locator: bool,
        ) -> dict[str, object]:
            locator: object
            if structured_locator:
                locator = {
                    "kind": "blue_archive.content_record.v1",
                    "pointer": "/content/0/TextCn",
                    "record_index": 0,
                }
            else:
                locator = "/content/0/TextCn"
            return {
                "atom_id": f"atom-{source_key[-6]}",
                "source_key": source_key,
                "record_locator": locator,
                "span_start": 0,
                "span_end": len(body),
                "source_file_sha256": source_hashes[source_key],
                "raw_span_sha256": _sha256_text(body),
                "review_status": "approved_for_scoring",
                "usable_for_scoring": True,
            }

        gold = {
            "schema_version": "aevnema.source-gold.v1",
            "status": "approved_for_scoring",
            "review_status": "approved_for_scoring",
            "usable_for_scoring": True,
            "source_manifest_sha256": _sha256_bytes(source_manifest_path),
            "families": [
                {
                    "family_id": "family-calibration",
                    "review_status": "approved_for_scoring",
                    "usable_for_scoring": True,
                    "claim_groups": [
                        {
                            "claim_group_id": "calibration-claim",
                            "review_status": "approved_for_scoring",
                            "usable_for_scoring": True,
                            "evidence_atoms": [
                                atom("main/a.json", _BODY_A, structured_locator=True)
                            ],
                        }
                    ],
                },
                {
                    "family_id": "family-holdout",
                    "review_status": "approved_for_scoring",
                    "usable_for_scoring": True,
                    "claim_groups": [
                        {
                            "claim_group_id": "holdout-claim",
                            "review_status": "approved_for_scoring",
                            "usable_for_scoring": True,
                            "evidence_atoms": [
                                atom("main/b.json", _BODY_B, structured_locator=False)
                            ],
                        }
                    ],
                },
            ],
        }
        gold_path = root / "gold.json"
        _write_json(gold_path, gold)
        split = {
            "schema_version": "aevnema.gold-split.v1",
            "status": "approved_for_scoring",
            "gold_manifest_sha256": _sha256_bytes(gold_path),
            "assignments": [
                {
                    "family_id": "family-calibration",
                    "split": "calibration",
                    "source_keys": ["main/a.json"],
                },
                {
                    "family_id": "family-holdout",
                    "split": "holdout",
                    "source_keys": ["main/b.json"],
                },
            ],
            "all_source_keys": ["main/a.json", "main/b.json"],
        }
        split_path = root / "split.json"
        _write_json(split_path, split)
        return {
            "source_root": source_root,
            "source_manifest_path": source_manifest_path,
            "gold_path": gold_path,
            "split_path": split_path,
            "gold": gold,
            "split": split,
            "source_hashes": source_hashes,
        }

    @staticmethod
    def _write_gold_and_rebind_split(fixture: dict[str, object]) -> None:
        gold_path = fixture["gold_path"]
        split_path = fixture["split_path"]
        assert isinstance(gold_path, Path)
        assert isinstance(split_path, Path)
        _write_json(gold_path, fixture["gold"])
        split = fixture["split"]
        assert isinstance(split, dict)
        split["gold_manifest_sha256"] = _sha256_bytes(gold_path)
        _write_json(split_path, split)

    @staticmethod
    def _validate(fixture: dict[str, object]) -> dict[str, object]:
        return validate_source_gold_files(
            fixture["gold_path"],  # type: ignore[arg-type]
            fixture["split_path"],  # type: ignore[arg-type]
            source_root=fixture["source_root"],  # type: ignore[arg-type]
            frozen_source_manifest_path=fixture["source_manifest_path"],  # type: ignore[arg-type]
        )

    @staticmethod
    def _cli_args(fixture: dict[str, object], output: Path) -> list[str]:
        return [
            "--gold",
            str(fixture["gold_path"]),
            "--split",
            str(fixture["split_path"]),
            "--source-root",
            str(fixture["source_root"]),
            "--frozen-source-manifest",
            str(fixture["source_manifest_path"]),
            "--output",
            str(output),
        ]

    def test_approved_source_gold_is_ready_and_report_never_exposes_source_text(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            report = self._validate(fixture)

        self.assertEqual(SOURCE_GOLD_VALIDATION_SCHEMA, report["schema"])
        self.assertEqual("ready", report["status"])
        self.assertTrue(report["scoring_eligible"])
        self.assertEqual([], report["issues"])
        self.assertEqual(
            {
                "families": 2,
                "claim_groups": 2,
                "evidence_atoms": 2,
                "frozen_sources": 2,
            },
            report["counts"],
        )
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(_BODY_A, rendered)
        self.assertNotIn(_BODY_B, rendered)

    def test_cli_writes_the_same_text_safe_local_report(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            output = root / "source-gold-report.json"
            exit_code = main(
                [
                    "--gold",
                    str(fixture["gold_path"]),
                    "--split",
                    str(fixture["split_path"]),
                    "--source-root",
                    str(fixture["source_root"]),
                    "--frozen-source-manifest",
                    str(fixture["source_manifest_path"]),
                    "--output",
                    str(output),
                ]
            )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertTrue(report["scoring_eligible"])
        self.assertNotIn(_BODY_A, json.dumps(report, ensure_ascii=False))

    def test_cli_refuses_report_output_over_artifacts_or_source_tree(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            artifact_paths = (
                fixture["gold_path"],
                fixture["split_path"],
                fixture["source_manifest_path"],
            )
            for artifact_path in artifact_paths:
                with self.subTest(output="artifact"):
                    assert isinstance(artifact_path, Path)
                    original = artifact_path.read_bytes()
                    self.assertEqual(2, main(self._cli_args(fixture, artifact_path)))
                    self.assertEqual(original, artifact_path.read_bytes())

            source_root = fixture["source_root"]
            assert isinstance(source_root, Path)
            source_output = source_root / "report.json"
            self.assertFalse(source_output.exists())
            self.assertEqual(2, main(self._cli_args(fixture, source_output)))
            self.assertFalse(source_output.exists())

    def test_cli_refuses_hard_link_output_alias_to_an_artifact(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            frozen_manifest = fixture["source_manifest_path"]
            assert isinstance(frozen_manifest, Path)
            alias = root / "artifact-alias.json"
            try:
                os.link(frozen_manifest, alias)
            except OSError as exc:  # pragma: no cover - filesystem policy dependent
                self.skipTest(f"hard links unavailable for this local filesystem: {exc}")
            original = frozen_manifest.read_bytes()

            self.assertEqual(2, main(self._cli_args(fixture, alias)))
            self.assertEqual(original, frozen_manifest.read_bytes())
            self.assertEqual(original, alias.read_bytes())

    def test_only_exact_bytes_entry_can_authorize_scoring(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            gold_path = fixture["gold_path"]
            split_path = fixture["split_path"]
            source_manifest_path = fixture["source_manifest_path"]
            assert isinstance(gold_path, Path)
            assert isinstance(split_path, Path)
            assert isinstance(source_manifest_path, Path)
            gold_bytes = gold_path.read_bytes()
            split_bytes = split_path.read_bytes()
            source_manifest_bytes = source_manifest_path.read_bytes()
            in_memory = validate_source_gold(
                json.loads(gold_bytes),
                json.loads(split_bytes),
                source_root=fixture["source_root"],  # type: ignore[arg-type]
                frozen_source_manifest=json.loads(source_manifest_bytes),
                frozen_source_manifest_sha256=sha256(source_manifest_bytes).hexdigest(),
                gold_manifest_sha256=sha256(gold_bytes).hexdigest(),
                split_manifest_sha256=sha256(split_bytes).hexdigest(),
            )
            from_bytes = validate_source_gold_bytes(
                gold_bytes,
                split_bytes,
                source_root=fixture["source_root"],  # type: ignore[arg-type]
                frozen_source_manifest_bytes=source_manifest_bytes,
            )

        self.assertFalse(in_memory["scoring_eligible"])
        self.assertIn("raw_artifact_binding_unavailable", _codes(in_memory))
        self.assertTrue(from_bytes["scoring_eligible"])

    def test_approved_atoms_fail_closed_for_required_binding_fields_and_span_errors(self) -> None:
        cases: tuple[tuple[str, str, object], ...] = (
            ("source_key", "atom_source_key_invalid", None),
            ("source_key", "atom_source_key_invalid", "C:/escaped.json"),
            ("source_key", "atom_source_key_invalid", "../escaped.json"),
            ("source_key", "atom_source_key_invalid", "MAIN/A.JSON"),
            ("source_key", "atom_source_key_invalid", "main/a.json."),
            ("source_key", "atom_source_key_invalid", "main/nul.json"),
            ("record_locator", "atom_record_locator_invalid", None),
            ("record_locator", "atom_record_locator_invalid", ""),
            ("span_start", "atom_span_not_integer", "0"),
            ("span_end", "atom_span_out_of_bounds", len(_BODY_A) + 1),
            ("source_file_sha256", "atom_source_file_sha256_missing", None),
            ("raw_span_sha256", "atom_raw_span_sha256_mismatch", "0" * 64),
        )
        for field, expected_code, value in cases:
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary))
                gold = fixture["gold"]
                assert isinstance(gold, dict)
                atom = gold["families"][0]["claim_groups"][0]["evidence_atoms"][0]
                assert isinstance(atom, dict)
                atom[field] = value
                self._write_gold_and_rebind_split(fixture)

                report = self._validate(fixture)

                self.assertEqual("invalid", report["status"])
                self.assertFalse(report["scoring_eligible"])
                self.assertIn(expected_code, _codes(report))

    def test_approved_schema_and_review_gates_are_explicit_at_every_level(self) -> None:
        cases = (
            ("schema", "gold_schema_version_invalid"),
            ("gold", "gold_review_status_not_approved"),
            ("family", "gold_family_review_status_not_approved"),
            ("claim_group", "gold_claim_group_review_status_not_approved"),
            ("atom", "gold_atom_review_status_not_approved"),
        )
        for level, expected_code in cases:
            with self.subTest(level=level), TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary))
                gold = fixture["gold"]
                assert isinstance(gold, dict)
                if level == "schema":
                    gold["schema_version"] = "aevnema.source-gold.draft.v1"
                elif level == "gold":
                    del gold["review_status"]
                else:
                    family = gold["families"][0]
                    assert isinstance(family, dict)
                    if level == "family":
                        del family["review_status"]
                    else:
                        group = family["claim_groups"][0]
                        assert isinstance(group, dict)
                        if level == "claim_group":
                            del group["review_status"]
                        else:
                            atom = group["evidence_atoms"][0]
                            assert isinstance(atom, dict)
                            del atom["review_status"]
                self._write_gold_and_rebind_split(fixture)

                report = self._validate(fixture)

                self.assertFalse(report["scoring_eligible"])
                self.assertIn(expected_code, _codes(report))

    def test_source_outside_frozen_scope_and_changed_file_fail_closed(self) -> None:
        with self.subTest("out_of_scope"), TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            gold = fixture["gold"]
            assert isinstance(gold, dict)
            atom = gold["families"][0]["claim_groups"][0]["evidence_atoms"][0]
            assert isinstance(atom, dict)
            atom["source_key"] = "main/outside.json"
            self._write_gold_and_rebind_split(fixture)

            report = self._validate(fixture)

            self.assertFalse(report["scoring_eligible"])
            self.assertIn("atom_source_not_in_frozen_scope", _codes(report))

        with self.subTest("changed_source_bytes"), TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            _write_json(
                root / "sources" / "main" / "a.json",
                {"content": [{"TextCn": "X" * len(_BODY_A)}]},
            )

            report = self._validate(fixture)

            self.assertFalse(report["scoring_eligible"])
            self.assertIn("frozen_source_file_sha256_mismatch", _codes(report))

    def test_unreferenced_frozen_source_bytes_are_still_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            source_c = root / "sources" / "main" / "c.json"
            untouched_body = "SYNTHETIC_UNREFERENCED_SOURCE_DO_NOT_REPORT"
            _write_json(source_c, {"content": [{"TextCn": untouched_body}]})

            source_manifest_path = fixture["source_manifest_path"]
            assert isinstance(source_manifest_path, Path)
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            source_manifest["sources"].append(
                {
                    "source_key": "main/c.json",
                    "source_file_sha256": _sha256_bytes(source_c),
                }
            )
            _write_json(source_manifest_path, source_manifest)
            gold = fixture["gold"]
            assert isinstance(gold, dict)
            gold["source_manifest_sha256"] = _sha256_bytes(source_manifest_path)
            self._write_gold_and_rebind_split(fixture)

            _write_json(
                source_c,
                {"content": [{"TextCn": "SYNTHETIC_TAMPERED_UNREFERENCED"}]},
            )
            report = self._validate(fixture)

        self.assertFalse(report["scoring_eligible"])
        self.assertIn("frozen_source_file_sha256_mismatch", _codes(report))
        self.assertNotIn(untouched_body, json.dumps(report, ensure_ascii=False))

    def test_source_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        for name, raw in (
            ("duplicate_keys", b'{"content": [], "content": []}'),
            ("nonfinite", b'{"content": NaN}'),
            ("overflowed_float", b'{"content": 1e999}'),
            ("too_deep", b'{"content": ' + (b"[" * 6000) + b"0" + (b"]" * 6000) + b"}"),
        ):
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                (root / "sources" / "main" / "a.json").write_bytes(raw)

                report = self._validate(fixture)

                self.assertFalse(report["scoring_eligible"])
                self.assertIn("frozen_source_json_unreadable", _codes(report))
                self.assertNotIn(_BODY_A, json.dumps(report, ensure_ascii=False))

    def test_artifact_json_rejects_overflowed_floats_and_excessive_depth(self) -> None:
        cases = (
            b'{"value": 1e999}',
            (b"[" * 6000) + b"0" + (b"]" * 6000),
        )
        for raw_gold in cases:
            with self.subTest(raw_gold_kind="overflow" if b"1e999" in raw_gold else "depth"), TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary))
                split_path = fixture["split_path"]
                source_manifest_path = fixture["source_manifest_path"]
                assert isinstance(split_path, Path)
                assert isinstance(source_manifest_path, Path)

                with self.assertRaises(SourceGoldValidationInputError):
                    validate_source_gold_bytes(
                        raw_gold,
                        split_path.read_bytes(),
                        source_root=fixture["source_root"],  # type: ignore[arg-type]
                        frozen_source_manifest_bytes=source_manifest_path.read_bytes(),
                    )

    def test_source_root_rejects_unc_syntax_and_link_aliases(self) -> None:
        unc_source_root = r"\\server\share\frozen-source"
        self.assertEqual(
            "source_root_unc_not_allowed",
            _source_root_rejection_code(unc_source_root),
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            with self.assertRaises(SourceGoldValidationInputError):
                validate_source_gold_files(
                    fixture["gold_path"],  # type: ignore[arg-type]
                    fixture["split_path"],  # type: ignore[arg-type]
                    source_root=unc_source_root,
                    frozen_source_manifest_path=fixture["source_manifest_path"],  # type: ignore[arg-type]
                )

            source_root = fixture["source_root"]
            assert isinstance(source_root, Path)
            source_root_alias = root / "source-root-alias"
            try:
                os.symlink(source_root, source_root_alias, target_is_directory=True)
            except OSError:
                return  # UNC rejection above is portable; symlink creation is policy-dependent.
            with self.assertRaises(SourceGoldValidationInputError):
                validate_source_gold_files(
                    fixture["gold_path"],  # type: ignore[arg-type]
                    fixture["split_path"],  # type: ignore[arg-type]
                    source_root=source_root_alias,
                    frozen_source_manifest_path=fixture["source_manifest_path"],  # type: ignore[arg-type]
                )
            self.assertEqual(
                "source_root_link_or_junction_not_allowed",
                _source_root_rejection_code(source_root_alias),
            )

    def test_artifact_inputs_reject_unc_before_file_api_or_cli_resolves_them(self) -> None:
        """Synthetic UNC syntax must not cause an input path to be opened."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            for label, argument in (
                ("gold", "--gold"),
                ("split", "--split"),
                ("frozen_source_manifest", "--frozen-source-manifest"),
            ):
                with self.subTest(label=label):
                    unc_artifact = rf"\\server\share\{label}.json"
                    self.assertEqual(
                        f"{label}_input_unc_not_allowed",
                        _artifact_input_rejection_code(unc_artifact, label=label),
                    )
                    file_inputs: dict[str, str | Path] = {
                        "gold": fixture["gold_path"],  # type: ignore[dict-item]
                        "split": fixture["split_path"],  # type: ignore[dict-item]
                        "frozen_source_manifest": fixture["source_manifest_path"],  # type: ignore[dict-item]
                    }
                    file_inputs[label] = unc_artifact
                    with self.assertRaises(SourceGoldValidationInputError):
                        validate_source_gold_files(
                            file_inputs["gold"],
                            file_inputs["split"],
                            source_root=fixture["source_root"],  # type: ignore[arg-type]
                            frozen_source_manifest_path=file_inputs[
                                "frozen_source_manifest"
                            ],
                        )

                    output = root / f"must-not-be-created-{label}.json"
                    cli_args = self._cli_args(fixture, output)
                    cli_args[cli_args.index(argument) + 1] = unc_artifact
                    self.assertEqual(2, main(cli_args))
                    self.assertFalse(output.exists())

    def test_artifact_inputs_reject_file_and_parent_symlink_aliases(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            gold_path = fixture["gold_path"]
            assert isinstance(gold_path, Path)
            final_alias = root / "gold-file-alias.json"
            parent_alias = root / "gold-parent-alias"
            try:
                os.symlink(gold_path, final_alias)
                os.symlink(root, parent_alias, target_is_directory=True)
            except OSError as exc:  # pragma: no cover - Windows privilege policy dependent
                self.skipTest(f"symlink creation unavailable for this local filesystem: {exc}")

            for alias in (final_alias, parent_alias / gold_path.name):
                with self.subTest(alias_kind="final" if alias == final_alias else "parent"):
                    self.assertEqual(
                        "gold_input_link_or_junction_not_allowed",
                        _artifact_input_rejection_code(alias, label="gold"),
                    )
                    with self.assertRaises(SourceGoldValidationInputError):
                        validate_source_gold_files(
                            alias,
                            fixture["split_path"],  # type: ignore[arg-type]
                            source_root=fixture["source_root"],  # type: ignore[arg-type]
                            frozen_source_manifest_path=fixture["source_manifest_path"],  # type: ignore[arg-type]
                        )

    def test_link_components_are_checked_root_to_leaf_before_the_leaf_can_be_opened(self) -> None:
        """A parent junction must stop validation before its child is lstat'ed."""

        with TemporaryDirectory() as temporary:
            leaf = Path(temporary) / "parent-junction" / "gold.json"
            parent = leaf.parent
            checked: list[Path] = []

            def simulated_link_check(candidate: Path) -> bool:
                checked.append(candidate)
                if candidate == leaf:
                    self.fail("the leaf must not be checked through a parent junction")
                return candidate == parent

            with patch(
                "benchmarks.validate_source_gold._is_link_or_junction",
                side_effect=simulated_link_check,
            ):
                self.assertTrue(_has_link_or_junction_component(leaf))
            self.assertIn(parent, checked)
            self.assertNotIn(leaf, checked)

    def test_relative_artifact_paths_are_fixed_before_a_cwd_change(self) -> None:
        """Reads continue using the verified absolute paths after CWD changes."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            other_directory = root / "other-working-directory"
            other_directory.mkdir()
            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                real_read_json_bytes = source_gold_validator._read_json_bytes
                seen_paths: list[Path] = []
                cwd_switched = False

                def switch_cwd_then_read(path: Path, *, label: str) -> bytes:
                    nonlocal cwd_switched
                    seen_paths.append(path)
                    if not cwd_switched:
                        cwd_switched = True
                        os.chdir(other_directory)
                    return real_read_json_bytes(path, label=label)

                with patch(
                    "benchmarks.validate_source_gold._read_json_bytes",
                    side_effect=switch_cwd_then_read,
                ):
                    report = validate_source_gold_files(
                        Path("gold.json"),
                        Path("split.json"),
                        source_root=Path("sources"),
                        frozen_source_manifest_path=Path("frozen-source-manifest.json"),
                    )
            finally:
                os.chdir(original_cwd)

        self.assertTrue(report["scoring_eligible"])
        self.assertEqual(3, len(seen_paths))
        self.assertTrue(all(path.is_absolute() for path in seen_paths))

    def test_windows_locality_policy_is_fail_closed_when_drive_probe_is_not_reliable(self) -> None:
        """Mocked drive classifications exercise no real mapped/network path."""

        remote_gold = r"Z:\remote\gold.json"
        local_gold = r"C:\local\gold.json"
        with patch("benchmarks.validate_source_gold.os.name", "nt"):
            with patch(
                "benchmarks.validate_source_gold._windows_get_drive_type",
                return_value=4,
            ):
                self.assertEqual(
                    "gold_input_remote_drive_not_allowed",
                    _artifact_input_rejection_code(remote_gold, label="gold"),
                )
            with patch(
                "benchmarks.validate_source_gold._windows_get_drive_type",
                return_value=None,
            ):
                self.assertEqual(
                    "gold_input_locality_unverifiable",
                    _artifact_input_rejection_code(local_gold, label="gold"),
                )
                self.assertEqual(
                    "source_root_locality_unverifiable",
                    _source_root_rejection_code(r"C:\local\sources"),
                )
            with patch(
                "benchmarks.validate_source_gold._windows_get_drive_type",
                return_value=3,
            ):
                self.assertIsNone(
                    _windows_locality_rejection_code(local_gold, prefix="gold_input")
                )

            with patch(
                "benchmarks.validate_source_gold.ctypes.windll",
                create=True,
            ) as windll:
                windll.kernel32.GetDriveTypeW.side_effect = OSError("synthetic denied")
                self.assertEqual(
                    "gold_input_locality_unverifiable",
                    _artifact_input_rejection_code(local_gold, label="gold"),
                )

        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            gold_path = fixture["gold_path"]
            assert isinstance(gold_path, Path)
            self.assertIsNone(_artifact_input_rejection_code(gold_path, label="gold"))

    def test_report_output_requires_an_existing_direct_local_parent(self) -> None:
        """No report path may create or resolve an unverified parent directory."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            unc_output = r"\\server\share\source-gold-report.json"
            self.assertEqual("output_unc_not_allowed", _report_output_rejection_code(unc_output))

            cli_args = self._cli_args(fixture, root / "unused.json")
            cli_args[cli_args.index("--output") + 1] = unc_output
            self.assertEqual(2, main(cli_args))

            missing_output = root / "missing-output-parent" / "report.json"
            self.assertEqual(2, main(self._cli_args(fixture, missing_output)))
            self.assertFalse(missing_output.parent.exists())

            existing_output = root / "existing-report.json"
            existing_output.write_text("old report", encoding="utf-8")
            self.assertIsNone(_report_output_rejection_code(existing_output))
            self.assertEqual(0, main(self._cli_args(fixture, existing_output)))
            self.assertTrue(json.loads(existing_output.read_text(encoding="utf-8"))["scoring_eligible"])

            output_parent_alias = root / "output-parent-alias"
            try:
                os.symlink(root, output_parent_alias, target_is_directory=True)
            except OSError:
                return  # pragma: no cover - Windows privilege policy dependent
            symlink_output = output_parent_alias / "report.json"
            self.assertEqual(
                "output_link_or_junction_not_allowed",
                _report_output_rejection_code(symlink_output),
            )
            self.assertEqual(2, main(self._cli_args(fixture, symlink_output)))
            self.assertFalse(symlink_output.exists())

    def test_relative_output_path_stays_bound_after_a_cwd_change(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            other_directory = root / "other-working-directory"
            other_directory.mkdir()
            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                real_named_temporary_file = source_gold_validator.NamedTemporaryFile
                cwd_switched = False

                def switch_cwd_then_create_temporary(*args: object, **kwargs: object) -> object:
                    nonlocal cwd_switched
                    if not cwd_switched:
                        cwd_switched = True
                        os.chdir(other_directory)
                    return real_named_temporary_file(*args, **kwargs)

                with patch(
                    "benchmarks.validate_source_gold.NamedTemporaryFile",
                    side_effect=switch_cwd_then_create_temporary,
                ):
                    exit_code = main(
                        [
                            "--gold",
                            "gold.json",
                            "--split",
                            "split.json",
                            "--source-root",
                            str(fixture["source_root"]),
                            "--frozen-source-manifest",
                            "frozen-source-manifest.json",
                            "--output",
                            "report.json",
                        ]
                    )
            finally:
                os.chdir(original_cwd)
            report_in_root = (root / "report.json").exists()
            report_in_other_directory = (other_directory / "report.json").exists()

        self.assertEqual(0, exit_code)
        self.assertTrue(report_in_root)
        self.assertFalse(report_in_other_directory)

    def test_cli_passes_verified_absolute_inputs_after_a_cwd_change(self) -> None:
        """The CLI must not discard the verified paths before validation."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            other_directory = root / "other-working-directory"
            other_directory.mkdir()
            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                real_validate_source_gold_files = source_gold_validator.validate_source_gold_files
                received_paths: list[Path] = []

                def switch_cwd_then_validate(
                    gold_path: Path,
                    split_path: Path,
                    *,
                    source_root: Path,
                    frozen_source_manifest_path: Path,
                ) -> dict[str, object]:
                    received_paths.extend(
                        (
                            gold_path,
                            split_path,
                            source_root,
                            frozen_source_manifest_path,
                        )
                    )
                    os.chdir(other_directory)
                    return real_validate_source_gold_files(
                        gold_path,
                        split_path,
                        source_root=source_root,
                        frozen_source_manifest_path=frozen_source_manifest_path,
                    )

                with patch(
                    "benchmarks.validate_source_gold.validate_source_gold_files",
                    side_effect=switch_cwd_then_validate,
                ):
                    exit_code = main(
                        [
                            "--gold",
                            "gold.json",
                            "--split",
                            "split.json",
                            "--source-root",
                            "sources",
                            "--frozen-source-manifest",
                            "frozen-source-manifest.json",
                            "--output",
                            "cli-report.json",
                        ]
                    )
            finally:
                os.chdir(original_cwd)
            report_in_root = (root / "cli-report.json").exists()
            report_in_other_directory = (other_directory / "cli-report.json").exists()

        self.assertEqual(0, exit_code)
        self.assertEqual(4, len(received_paths))
        self.assertTrue(all(path.is_absolute() for path in received_paths))
        self.assertTrue(report_in_root)
        self.assertFalse(report_in_other_directory)

    def test_windows_output_locality_probe_is_fail_closed_without_network_access(self) -> None:
        remote_output = r"Z:\remote\report.json"
        local_output = r"C:\local\report.json"
        with patch("benchmarks.validate_source_gold.os.name", "nt"):
            with patch(
                "benchmarks.validate_source_gold._windows_get_drive_type",
                return_value=4,
            ):
                self.assertEqual(
                    "output_remote_drive_not_allowed",
                    _report_output_rejection_code(remote_output),
                )
            with patch(
                "benchmarks.validate_source_gold._windows_get_drive_type",
                return_value=None,
            ):
                self.assertEqual(
                    "output_locality_unverifiable",
                    _report_output_rejection_code(local_output),
                )

    def test_frozen_manifest_rejects_physical_source_aliases(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            original = root / "sources" / "main" / "a.json"
            alias = root / "sources" / "main" / "alias.json"
            try:
                os.link(original, alias)
            except OSError as exc:  # pragma: no cover - unusual filesystem policy
                self.skipTest(f"hard links unavailable for this local filesystem: {exc}")
            source_manifest_path = fixture["source_manifest_path"]
            assert isinstance(source_manifest_path, Path)
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            source_manifest["sources"].append(
                {
                    "source_key": "main/alias.json",
                    "source_file_sha256": _sha256_bytes(alias),
                }
            )
            _write_json(source_manifest_path, source_manifest)
            gold = fixture["gold"]
            assert isinstance(gold, dict)
            gold["source_manifest_sha256"] = _sha256_bytes(source_manifest_path)
            self._write_gold_and_rebind_split(fixture)

            report = self._validate(fixture)

        self.assertFalse(report["scoring_eligible"])
        self.assertIn("frozen_source_physical_alias", _codes(report))

    def test_oversized_json_pointer_index_fails_closed_without_crashing(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            gold = fixture["gold"]
            assert isinstance(gold, dict)
            atom = gold["families"][0]["claim_groups"][0]["evidence_atoms"][0]
            assert isinstance(atom, dict)
            atom["record_locator"] = "/content/" + ("9" * 5000)
            self._write_gold_and_rebind_split(fixture)

            report = self._validate(fixture)

        self.assertFalse(report["scoring_eligible"])
        self.assertIn("atom_record_locator_unresolved", _codes(report))

    def test_source_overlap_and_absent_holdout_fail_closed(self) -> None:
        with self.subTest("holdout_overlap"), TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            gold = fixture["gold"]
            split = fixture["split"]
            assert isinstance(gold, dict)
            assert isinstance(split, dict)
            source_atom = deepcopy(gold["families"][0]["claim_groups"][0]["evidence_atoms"][0])
            gold["families"][1]["claim_groups"][0]["evidence_atoms"] = [source_atom]
            split["assignments"][1]["source_keys"] = ["main/a.json"]
            split["all_source_keys"] = ["main/a.json"]
            self._write_gold_and_rebind_split(fixture)

            report = self._validate(fixture)

            self.assertFalse(report["scoring_eligible"])
            self.assertIn("holdout_source_overlap", _codes(report))

        with self.subTest("missing_holdout"), TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            split = fixture["split"]
            assert isinstance(split, dict)
            split["assignments"][1]["split"] = "calibration"
            self._write_gold_and_rebind_split(fixture)

            report = self._validate(fixture)

            self.assertFalse(report["scoring_eligible"])
            self.assertIn("split_missing_holdout", _codes(report))

    def test_draft_is_diagnosable_but_never_scoring_eligible(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            gold = fixture["gold"]
            split = fixture["split"]
            assert isinstance(gold, dict)
            assert isinstance(split, dict)
            gold["status"] = "draft_pending_source_span_review"
            gold["usable_for_scoring"] = False
            for family in gold["families"]:
                family["usable_for_scoring"] = False
                for group in family["claim_groups"]:
                    group["usable_for_scoring"] = False
                    for atom in group["evidence_atoms"]:
                        atom["usable_for_scoring"] = False
            atom = gold["families"][0]["claim_groups"][0]["evidence_atoms"][0]
            atom["record_locator"] = None
            atom["span_start"] = None
            atom["span_end"] = None
            atom["raw_span_sha256"] = None
            split["status"] = "draft_pending_source_span_review"
            self._write_gold_and_rebind_split(fixture)

            report = self._validate(fixture)

        self.assertEqual("draft", report["status"])
        self.assertFalse(report["scoring_eligible"])
        self.assertEqual([], report["issues"])
        self.assertIn("gold_draft_not_scorable", _codes(report, "diagnostics"))
        self.assertIn("atom_record_locator_invalid", _codes(report, "diagnostics"))

    def test_draft_schema_cannot_be_promoted_only_by_status_and_flags(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            gold = fixture["gold"]
            split = fixture["split"]
            assert isinstance(gold, dict)
            assert isinstance(split, dict)
            gold["schema_version"] = "aevnema.source-gold.draft.v1"
            split["schema_version"] = "aevnema.gold-split.draft.v1"
            self._write_gold_and_rebind_split(fixture)

            report = self._validate(fixture)

        self.assertFalse(report["scoring_eligible"])
        self.assertIn("gold_schema_version_invalid", _codes(report))
        self.assertIn("split_schema_version_invalid", _codes(report))

    def test_report_does_not_reflect_untrusted_manifest_values(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            gold = fixture["gold"]
            split = fixture["split"]
            assert isinstance(gold, dict)
            assert isinstance(split, dict)
            gold["status"] = _BODY_A
            split["status"] = _BODY_B
            self._write_gold_and_rebind_split(fixture)

            report = self._validate(fixture)

        rendered = json.dumps(report, ensure_ascii=False)
        self.assertFalse(report["scoring_eligible"])
        self.assertNotIn(_BODY_A, rendered)
        self.assertNotIn(_BODY_B, rendered)


if __name__ == "__main__":
    unittest.main()
