"""PTS-based sampling, passage windows, and discontinuity detection.

Never use CAP_PROP_FPS or wall-clock arrival time for sampling, dwell, or speed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PtsSampler:
    interval_ms: float
    last_taken_ms: float | None = None

    def should_take(self, pts_ms: float | None) -> bool:
        if pts_ms is None:
            return True
        if self.last_taken_ms is None:
            self.last_taken_ms = pts_ms
            return True
        if pts_ms - self.last_taken_ms + 1e-9 >= self.interval_ms:
            self.last_taken_ms = pts_ms
            return True
        return False

    def reset(self) -> None:
        self.last_taken_ms = None


@dataclass
class PassageClock:
    gap_ms: float
    jump_ms: float
    last_pts_ms: float | None = None
    passage_serial: int = 0
    events: list[str] = field(default_factory=list)

    def observe(self, pts_ms: float | None) -> str:
        """Return 'ok' or 'reset'. Irregular gaps never raise."""
        if pts_ms is None:
            return "ok"
        prev = self.last_pts_ms
        self.last_pts_ms = pts_ms
        if prev is None:
            return "ok"
        delta = pts_ms - prev
        if delta < 0:
            self.events.append("pts_regression")
            self._bump()
            return "reset"
        if self.jump_ms > 0 and delta > self.jump_ms:
            self.events.append("pts_jump")
            self._bump()
            return "reset"
        if self.gap_ms > 0 and delta > self.gap_ms:
            self.events.append("passage_gap")
            self._bump()
            return "reset"
        return "ok"

    def _bump(self) -> None:
        self.passage_serial += 1

    def reset(self) -> None:
        self.last_pts_ms = None
        self.passage_serial += 1
        self.events.append("manual_reset")


def backoff_seconds(attempt: int, start: float = 2.0, cap: float = 30.0) -> float:
    """Bounded exponential backoff. attempt is 1-based. Never zero."""
    n = max(1, int(attempt))
    delay = start * (2 ** (n - 1))
    return float(min(cap, max(start, delay)))


def source_time_from_ingest(ingest_utc, clock_offset_ms: float | None):
    """Do not fabricate a camera clock. Apply offset only when known."""
    if clock_offset_ms is None:
        return ingest_utc, False
    from datetime import timedelta

    return ingest_utc + timedelta(milliseconds=clock_offset_ms), True
