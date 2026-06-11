from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from mmbt.core.types import BookLevel, MarketTick, OrderBook, Side, Trade
from mmbt.data.synthetic import SyntheticConfig, generate_ticks

_REQUIRED = {"ts", "bid_px", "bid_sz", "ask_px", "ask_sz"}
_TRADE    = {"trade_px", "trade_sz", "trade_side"}


class TickLoader:
    """
    Lazy tick iterator. Wraps any source — CSV, Parquet, synthetic.
    Use to_list() only when you know the dataset fits in RAM.
    """

    def __init__(self, source: Callable[[], Iterator[MarketTick]]) -> None:
        self._source = source

    def __iter__(self) -> Iterator[MarketTick]:
        return self._source()

    def to_list(self) -> list[MarketTick]:
        return list(self._source())

    def head(self, n: int) -> list[MarketTick]:
        result = []
        for tick in self._source():
            result.append(tick)
            if len(result) >= n:
                break
        return result

    @classmethod
    def from_csv(cls, path: str | Path, symbol: str = "UNKNOWN", chunk_size: int = 10_000) -> TickLoader:
        p = Path(path)
        return cls(lambda: _csv_iter(p, symbol, chunk_size))

    @classmethod
    def from_parquet(cls, path: str | Path, symbol: str = "UNKNOWN") -> TickLoader:
        p = Path(path)
        return cls(lambda: _parquet_iter(p, symbol))

    @classmethod
    def synthetic(cls, config: SyntheticConfig) -> TickLoader:
        return cls(lambda: generate_ticks(config))


def _csv_iter(path: Path, symbol: str, chunk_size: int) -> Iterator[MarketTick]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install mmbt[dev]")

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        missing = _REQUIRED - set(chunk.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        for _, row in chunk.iterrows():
            yield _row_to_tick(row, symbol)


def _parquet_iter(path: Path, symbol: str) -> Iterator[MarketTick]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install mmbt[dev]")

    # loads full file — TODO: row-group streaming for large Parquet files
    df = pd.read_parquet(path)
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")
    for _, row in df.iterrows():
        yield _row_to_tick(row, symbol)


def _row_to_tick(row: object, symbol: str) -> MarketTick:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install realistic-mm-backtester[dev]")

    ts   = float(row["ts"])
    book = OrderBook(
        bids=[BookLevel(float(row["bid_px"]), float(row["bid_sz"]))],
        asks=[BookLevel(float(row["ask_px"]), float(row["ask_sz"]))],
        ts=ts,
    )
    trades: list[Trade] = []
    if _TRADE.issubset(row.index) and all(pd.notna(row[c]) for c in ("trade_px", "trade_sz", "trade_side")):
        side_str = str(row["trade_side"]).strip().upper()
        if side_str not in ("BUY", "SELL"):
            raise ValueError(f"trade_side must be BUY or SELL, got '{side_str}'")
        trades.append(Trade(
            price=float(row["trade_px"]),
            size=float(row["trade_sz"]),
            side=Side.BUY if side_str == "BUY" else Side.SELL,
            ts=ts,
            is_liquidation=bool(row.get("is_liquidation", False)),
        ))
    return MarketTick(book=book, trades=trades, ts=ts)
