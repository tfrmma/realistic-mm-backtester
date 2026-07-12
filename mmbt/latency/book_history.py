from __future__ import annotations

import bisect
from dataclasses import dataclass

from mmbt.core.types import OrderBook, Trade


@dataclass(slots=True)
class BookSnapshot:
    ts: float
    book: OrderBook
    trades: list[Trade]


class BookHistory:
    """
    Bounded ring buffer of recent (book, trades) snapshots, keyed by tick ts.

    Lets the engine hand a strategy the market view it would actually have
    seen `feed_us` after the fact, instead of the current tick's book —
    closing the gap noted in ProBacktestEngine's tick-flow docstring.

    maxlen bounds memory. If a sampled feed delay looks further back than the
    buffer currently holds (e.g. right at the start of a run, or an unusually
    large delay draw), `as_of` falls back to the oldest snapshot available —
    the same tradeoff a real feed subscriber makes when it falls behind and
    a ring buffer on the wire has already rolled over.
    """

    def __init__(self, maxlen: int = 2_000) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._maxlen  = maxlen
        self._ts: list[float] = []
        self._snaps: list[BookSnapshot] = []

    def push(self, ts: float, book: OrderBook, trades: list[Trade]) -> None:
        self._ts.append(ts)
        self._snaps.append(BookSnapshot(ts=ts, book=book, trades=trades))
        if len(self._ts) > self._maxlen:
            del self._ts[0]
            del self._snaps[0]

    def as_of(self, target_ts: float) -> BookSnapshot | None:
        """Most recent snapshot with ts <= target_ts. None only if nothing has
        been pushed yet."""
        if not self._ts:
            return None
        idx = bisect.bisect_right(self._ts, target_ts) - 1
        if idx < 0:
            return self._snaps[0]  # target predates all retained history
        return self._snaps[idx]

    def __len__(self) -> int:
        return len(self._ts)
