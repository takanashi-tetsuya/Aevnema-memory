"""Run the preserved socket-blocked suite into a fresh output directory."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "validation" / ("offline-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output = output.resolve()
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing test report; choose a fresh output directory.")
    script = ROOT / "validation/associative-recall-experiment-20260911/run_offline_regression.py"
    spec = importlib.util.spec_from_file_location("handoff_offline_checks", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUTPUT = output
    return module.main()

if __name__ == "__main__":
    raise SystemExit(main())
