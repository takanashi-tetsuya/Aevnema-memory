"""Public API for the associative-memory engine.

Chat applications should depend on this module and ``memory_demo.contracts``.
The ingestion, repository, retrieval and benchmark packages are implementation
details and may evolve independently.
"""

from .app import MemoryApplication
from .config import AppConfig
from .database import Database

__all__ = ["AppConfig", "Database", "MemoryApplication"]
