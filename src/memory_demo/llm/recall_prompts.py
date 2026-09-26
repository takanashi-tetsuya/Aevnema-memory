"""Load the recall catalog from the same workspace as the engine prompts."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_path = Path(__file__).resolve().parents[3] / "config" / "prompt_config" / "progressive_recall_prompts.py"
_spec = spec_from_file_location("memory_demo._recall_prompt_catalog", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load recall prompt catalog: {_path}")
_catalog = module_from_spec(_spec)
_spec.loader.exec_module(_catalog)
RECALL_PLAN_SYSTEM = _catalog.RECALL_PLAN_SYSTEM
RECALL_MAP_SYSTEM = _catalog.RECALL_MAP_SYSTEM
RECALL_VERIFY_SYSTEM = _catalog.RECALL_VERIFY_SYSTEM
