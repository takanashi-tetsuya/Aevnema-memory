"""Small request-local policies for progressive associative recall."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re


_MAXIMUM_REQUEST = re.compile(
    r"^\s*(?:(?:请|麻烦)(?:你)?\s*)?(?:尽最大努力|竭尽全力)(?:地)?(?:回想|回忆|搜索)"
    r"|^\s*(?:please\s+)?(?:try your (?:very )?best to recall|make (?:the )?maximum effort to recall)",
    re.IGNORECASE,
)


def requests_maximum_recall(question: str) -> bool:
    """Recognize an explicit imperative, not a mention inside quoted evidence."""
    return bool(_MAXIMUM_REQUEST.search(str(question)))


@dataclass(frozen=True, slots=True)
class RecallPolicy:
    mode: str
    timeout_seconds: float
    seeds_per_cue: int
    wave_expansions: int
    sources_per_review: int
    source_window_chars: int = 6000
    max_cues: int = 6
    max_needs: int = 12
    learning_enabled: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"light", "standard", "deep", "max_effort"}:
            raise ValueError("unknown recall mode")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("recall timeout must be positive and finite")
        for name in ("seeds_per_cue", "wave_expansions", "sources_per_review", "source_window_chars", "max_cues", "max_needs"):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def for_request(cls, question: str, mode: str | None = None, *, timeout_seconds: float | None = None, learn: bool = False) -> "RecallPolicy":
        selected = mode or ("max_effort" if requests_maximum_recall(question) else "deep")
        settings = {
            "light": (20.0, 2, 32, 2),
            "standard": (60.0, 4, 128, 3),
            "deep": (360.0, 8, 512, 4),
            "max_effort": (1800.0, 16, 2048, 6),
        }
        if selected not in settings:
            raise ValueError("mode must be light, standard, deep, or max_effort")
        seconds, seeds, wave, batch = settings[selected]
        policy = cls(selected, seconds, seeds, wave, batch, learning_enabled=learn)
        if timeout_seconds is not None:
            requested = float(timeout_seconds)
            if not math.isfinite(requested) or requested <= 0:
                raise ValueError("timeout_seconds must be positive and finite")
            # An explicit shorter deadline is authoritative. Escalation uses a
            # named mode, not an accidental unbounded float.
            policy = replace(policy, timeout_seconds=min(seconds, requested))
        return policy
