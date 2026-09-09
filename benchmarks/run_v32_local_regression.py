"""Run the fixed v3.2 local regression set and persist a receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys


TESTS = [
    "tests.test_query_learning_real_pipeline_v3",
    "tests.test_frozen_evidence_matrix",
    "tests.test_source_validated_control_cache",
    "tests.test_same_chapter_learning_matrix",
    "tests.test_q1_q2_diagnostic_pilot",
    "tests.test_contextual_exact_revisit_t15",
    "tests.test_model_and_evidence.ModelAndEvidenceTests.test_source_excerpt_delivers_raw_record_for_verified_reasoning_view_quote",
    "tests.test_model_and_evidence.ModelAndEvidenceTests.test_projected_evidence_rejects_duplicate_language_fields",
    "tests.test_model_and_evidence.ModelAndEvidenceTests.test_projected_evidence_rejects_multiline_translation_field",
    "tests.test_model_and_evidence.ModelAndEvidenceTests.test_answer_audit_prompt_receives_the_full_delivered_source_excerpt",
    "tests.test_answer_budget_admission_v31.AnswerBudgetAdmissionTests.test_insufficient_full_chain_budget_skips_correction_and_records_checkpoints",
]


def run(*, root: Path, output_dir: Path) -> dict[str, object]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONUTF8"] = "1"
    command = [sys.executable, "-m", "unittest", "-v", *TESTS]
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    stdout_path = output_dir / "E32-00_local_regression.stdout.txt"
    stdout_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    match = re.search(r"Ran (\d+) tests?", completed.stdout + completed.stderr)
    receipt = {
        "schema": "aevnema.v3_2.local_regression_receipt.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "tests": TESTS,
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "test_count": int(match.group(1)) if match else "not_observed",
        "stdout_artifact": str(stdout_path),
        "stdout_sha256": sha256(stdout_path.read_bytes()).hexdigest(),
        "network": "not_used_by_this_runner",
    }
    (output_dir / "E32-00_local_regression.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(root=args.root, output_dir=args.output_dir)
    print(json.dumps(receipt, ensure_ascii=False))
    raise SystemExit(int(receipt["returncode"]))


if __name__ == "__main__":
    main()
