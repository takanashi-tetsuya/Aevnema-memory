"""Compatibility entry point for the refactored importer."""

import sys

from memory_demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["import", *sys.argv[1:]]))
