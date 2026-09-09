"""Replay B/S/T/M evidence-only arms from one frozen N12b Q2 snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.run_v32_prepared_early_candidate_pool_matrix import run


CONDITIONS: tuple[tuple[str, bool, bool, bool], ...] = (
    ("B_ordinary_no_edge", False, True, False),
    ("S_prepared_observe_only", True, False, False),
    ("T_prepared_candidate_treatment", True, False, True),
    ("M_prepared_candidate_treatment_masked", True, True, True),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-snapshot", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--prepared-plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--treatment-edge-id", type=int, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                **vars(args),
                conditions=CONDITIONS,
                schema="aevnema.v3_2.prepared_early_bstm_matrix.v1",
                report_filename="R2_BSTM_EVIDENCE_ONLY_MATRIX.md",
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
