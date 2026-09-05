"""Stable import facade for the human-auditable prompt catalog.

Prompt text and prompt-building contracts live outside the runtime package in
``config/prompt_config/memory_prompts.py``.  Keeping this facade preserves the
engine's public imports while preventing model instructions from spreading
through business logic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_catalog() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "prompt_config"
        / "memory_prompts.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"memory prompt catalog not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "memory_demo._configured_prompts", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load memory prompt catalog: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_catalog = _load_catalog()
__all__ = tuple(_catalog.__all__)
globals().update({name: getattr(_catalog, name) for name in __all__})
