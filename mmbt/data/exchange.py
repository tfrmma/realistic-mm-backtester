from __future__ import annotations

from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Iterator

from mmbt.core.types import MarketTick
from mmbt.data.loader import TickLoader


@dataclass(frozen=True)
class ExchangeMetadata:
    tick_size: float       = 0.01
    min_order_size: float  = 0.001
    fee_rate_maker: float  = 0.0001
    fee_rate_taker: float  = 0.0005


class CSVExchange:
    """
    Exchange adapter backed by CSV/Parquet files.
    Implements the Exchange protocol. Swap for a live feed adapter in prod.

    Usage:
        ex = CSVExchange("hyperliquid")
        ex.register("BTC-USD", "data/btc.csv", ExchangeMetadata(tick_size=0.1))
        for tick in ex.load_ticks("BTC-USD"):
            ...
    """

    def __init__(self, name: str = "csv", default_metadata: ExchangeMetadata | None = None) -> None:
        self.name          = name
        self._paths: dict[str, Path] = {}
        self._meta: dict[str, ExchangeMetadata] = {}
        self._default      = default_metadata or ExchangeMetadata()

    def register(self, symbol: str, path: str | Path, meta: ExchangeMetadata | None = None) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"tick file not found: {p}")
        self._paths[symbol] = p
        if meta:
            self._meta[symbol] = meta

    def load_ticks(self, symbol: str, start_ts: float = -inf, end_ts: float = inf) -> Iterator[MarketTick]:
        path = self._paths.get(symbol)
        if path is None:
            raise ValueError(f"no file registered for '{symbol}'")
        loader = (
            TickLoader.from_parquet(path, symbol=symbol)
            if path.suffix == ".parquet"
            else TickLoader.from_csv(path, symbol=symbol)
        )
        for tick in loader:
            if tick.ts < start_ts:
                continue
            if tick.ts > end_ts:
                break
            yield tick

    def tick_size(self, symbol: str) -> float:
        return self._meta.get(symbol, self._default).tick_size

    def min_order_size(self, symbol: str) -> float:
        return self._meta.get(symbol, self._default).min_order_size

    def fee_rate_maker(self, symbol: str) -> float:
        return self._meta.get(symbol, self._default).fee_rate_maker

    def fee_rate_taker(self, symbol: str) -> float:
        return self._meta.get(symbol, self._default).fee_rate_taker

    def registered_symbols(self) -> list[str]:
        return list(self._paths.keys())

    def __repr__(self) -> str:
        return f"CSVExchange(name={self.name!r}, symbols={self.registered_symbols()})"
