"""Monotonic, request-scoped deadline accounting.

This module deliberately models only the client-side budget.  It cannot
promise that a remote provider stops computing after a synchronous HTTP call
has already crossed the network boundary; callers use it to stop queueing,
retrying, accepting late results, and performing later local side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


class DeadlineExpired(TimeoutError):
    """Raised when a caller-visible monotonic deadline has no time left."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"request deadline exceeded before {stage}")
        self.stage = str(stage)


@dataclass(frozen=True, slots=True)
class DeadlineBudget:
    """A single absolute deadline shared by all nested request stages."""

    deadline_at: float

    def __post_init__(self) -> None:
        value = float(self.deadline_at)
        if not math.isfinite(value):
            raise ValueError("deadline_at must be a finite monotonic timestamp")
        object.__setattr__(self, "deadline_at", value)

    @classmethod
    def from_timeout(cls, timeout_seconds: float) -> "DeadlineBudget":
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout):
            raise ValueError("timeout_seconds must be finite")
        return cls(time.monotonic() + max(0.0, timeout))

    def remaining(self) -> float:
        """Return non-negative remaining local wall-clock budget in seconds."""

        return max(0.0, float(self.deadline_at) - time.monotonic())

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def require(self, stage: str) -> float:
        """Return remaining seconds or fail before starting ``stage``."""

        remaining = self.remaining()
        if remaining <= 0.0:
            raise DeadlineExpired(stage)
        return remaining
