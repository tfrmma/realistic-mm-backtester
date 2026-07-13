from __future__ import annotations

import bisect
from pathlib import Path


class OpenInterestSchedule:
    """
    Standalone, engine-agnostic lookup of open interest over time. Not part
    of the tick stream or any engine OI typically updates far less often
    than the book does (seconds to minutes, not ticks), so it doesn't belong
    baked into MarketTick/OrderBook. Build one yourself and query it from
    inside your own Strategy.on_tick using the current book.ts:

        oi_sched = OpenInterestSchedule.from_csv("btc_oi.csv")

        class MyStrategy(BaseStrategy):
            def __init__(self, oi_sched: OpenInterestSchedule) -> None:
                self.oi_sched = oi_sched

            def on_tick(self, book, trades):
                oi = self.oi_sched.as_of(book.ts)
                ...

    No engine or protocol changes needed the schedule is just data your
    own strategy happens to hold a reference to.

    Real OI updates are seconds-to-minutes apart, not microseconds, so a full
    in-memory schedule stays small even for a long backtest unlike
    latency/book_history.py's BookHistory, there's no bounded-memory /
    ring-buffer tradeoff to make here.
    """

    def __init__(self, points: list[tuple[float, float]]) -> None:
        ordered = sorted(points, key=lambda p: p[0])
        self._ts: list[float] = [p[0] for p in ordered]
        self._oi: list[float] = [p[1] for p in ordered]

    @classmethod
    def from_csv(cls, path: str | Path, ts_col: str = "ts", oi_col: str = "oi") -> OpenInterestSchedule:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install mmbt[dev]")
        df = pd.read_csv(path)
        missing = {ts_col, oi_col} - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        return cls(list(zip(df[ts_col].astype(float), df[oi_col].astype(float))))

    @classmethod
    def from_dict(cls, mapping: dict[float, float]) -> OpenInterestSchedule:
        return cls(list(mapping.items()))

    def as_of(self, ts: float) -> float | None:
        """Most recent OI value at or before ts (forward-fill, since OI is a
        step function between updates). None if ts predates the earliest
        known point there's nothing to fill forward from."""
        if not self._ts:
            return None
        idx = bisect.bisect_right(self._ts, ts) - 1
        if idx < 0:
            return None
        return self._oi[idx]

    def change(self, ts: float, lookback_us: float) -> float | None:
        """as_of(ts) - as_of(ts - lookback_us). None if either side is
        unavailable (too close to the start of the schedule)."""
        now  = self.as_of(ts)
        past = self.as_of(ts - lookback_us)
        if now is None or past is None:
            return None
        return now - past

    def __len__(self) -> int:
        return len(self._ts)


def generate_oi_schedule(
    n_points: int = 1_000,
    start_oi: float = 10_000_000.0,
    interval_us: float = 60_000_000.0,  # 1 minute OI updates far slower than book ticks
    drift: float = 0.0,
    vol: float = 50_000.0,
    seed: int | None = None,
) -> OpenInterestSchedule:
    """
    Synthetic OI random walk, same spirit as data/synthetic.py's price
    generator not realistic, good enough to smoke-test a strategy that
    reads OI without waiting on real data. Floored at 0 (OI can't go negative).
    """
    import numpy as np

    rng    = np.random.default_rng(seed)
    oi     = start_oi
    ts     = 0.0
    points: list[tuple[float, float]] = []
    for _ in range(n_points):
        points.append((ts, oi))
        oi = max(0.0, oi + drift + float(rng.normal(0.0, vol)))
        ts += interval_us
    return OpenInterestSchedule(points)
