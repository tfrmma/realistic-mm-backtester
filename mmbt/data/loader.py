from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from mmbt.core.types import BookLevel, MarketTick, OrderBook, Side, Trade
from mmbt.data.synthetic import SyntheticConfig, generate_ticks

_REQUIRED = {"ts", "bid_px", "bid_sz", "ask_px", "ask_sz"}
_TRADE    = {"trade_px", "trade_sz", "trade_side"}


def _level_columns(columns: set[str], prefix: str) -> list[tuple[str, str]]:
    """
    Ordered (price_col, size_col) pairs for one side of the book, best level
    first. `{prefix}_px`/`{prefix}_sz` (no suffix) is level 1 -- kept bare for
    backward compatibility with single-level CSVs. Deeper levels are
    `{prefix}_px_2`/`{prefix}_sz_2`, `{prefix}_px_3`/`{prefix}_sz_3`, etc.,
    picked up in order until a number is missing. Provide as many as your
    data has (5-10 is typical for real queue simulation) -- one level still
    works exactly like before.
    """
    pairs: list[tuple[str, str]] = []
    if f"{prefix}_px" in columns and f"{prefix}_sz" in columns:
        pairs.append((f"{prefix}_px", f"{prefix}_sz"))
    n = 2
    while f"{prefix}_px_{n}" in columns and f"{prefix}_sz_{n}" in columns:
        pairs.append((f"{prefix}_px_{n}", f"{prefix}_sz_{n}"))
        n += 1
    return pairs


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

    bid_cols: list[tuple[str, str]] | None = None
    ask_cols: list[tuple[str, str]] | None = None
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        missing = _REQUIRED - set(chunk.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        if bid_cols is None:  # column set is stable across chunks of one file
            bid_cols = _level_columns(set(chunk.columns), "bid")
            ask_cols = _level_columns(set(chunk.columns), "ask")
        for _, row in chunk.iterrows():
            yield _row_to_tick(row, symbol, bid_cols, ask_cols)


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
    bid_cols = _level_columns(set(df.columns), "bid")
    ask_cols = _level_columns(set(df.columns), "ask")
    for _, row in df.iterrows():
        yield _row_to_tick(row, symbol, bid_cols, ask_cols)


def _row_to_tick(
    row: object,
    symbol: str,
    bid_cols: list[tuple[str, str]],
    ask_cols: list[tuple[str, str]],
) -> MarketTick:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install realistic-mm-backtester[dev]")

    ts   = float(row["ts"])
    book = OrderBook(
        bids=_levels(row, bid_cols, pd),
        asks=_levels(row, ask_cols, pd),
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


def _levels(row: object, cols: list[tuple[str, str]], pd) -> list[BookLevel]:
    # stop at the first missing/NaN level -- a shallower book on some rows
    # (thin period, exchange only sent N levels that tick) is normal, not an error
    levels: list[BookLevel] = []
    for px_col, sz_col in cols:
        px, sz = row[px_col], row[sz_col]
        if pd.isna(px) or pd.isna(sz):
            break
        levels.append(BookLevel(float(px), float(sz)))
    if not levels:
        raise ValueError(f"no valid book levels found in columns {cols}")
    return levels
