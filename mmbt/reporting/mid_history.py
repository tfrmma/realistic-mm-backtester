from __future__ import annotations

import numpy as np


class MidHistoryBuffer:
    """
    Fixed-capacity circular buffer of (ts, mid) points, backed by numpy arrays
    instead of a Python list that grows one tuple per tick forever.

    Bounds memory for very long backtests (multi-month / >10M ticks) at a
    real cost: once the buffer wraps, only the most recent `capacity` points
    are retained. BacktestReport's adverse-selection lookback can only look
    forward from a fill if that fill's tick is still inside the retained
    window -- fills older than that are silently skipped, same as fills too
    close to the very end of a run already were before this existed. Size
    `capacity` to your expected run length if you need full adverse-selection
    coverage; the default (200k points, ~3MB) covers most single backtests
    without eating memory across a parallel sweep's many worker processes.
    """

    def __init__(self, capacity: int = 200_000) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._ts    = np.empty(capacity, dtype=np.float64)
        self._mid   = np.empty(capacity, dtype=np.float64)
        self._write = 0   # next write slot, mod capacity
        self._count = 0   # valid entries, capped at capacity

    def append(self, point: tuple[float, float]) -> None:
        ts, mid = point
        self._ts[self._write]  = ts
        self._mid[self._write] = mid
        self._write = (self._write + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)

    def __len__(self) -> int:
        return self._count

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Chronologically-ordered (ts, mid) numpy arrays of the retained window."""
        if self._count < self._capacity:
            return self._ts[:self._count].copy(), self._mid[:self._count].copy()
        # buffer has wrapped at least once oldest entry sits at _write
        ts  = np.concatenate([self._ts[self._write:],  self._ts[:self._write]])
        mid = np.concatenate([self._mid[self._write:], self._mid[:self._write]])
        return ts, mid

    def __iter__(self):
        ts, mid = self.to_arrays()
        return iter(zip(ts.tolist(), mid.tolist()))
