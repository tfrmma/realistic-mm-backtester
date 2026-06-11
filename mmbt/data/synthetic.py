from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from mmbt.core.types import BookLevel, MarketTick, OrderBook, Side, Trade


@dataclass
class SyntheticConfig:
    n_ticks: int          = 10_000
    start_price: float    = 50_000.0
    vol_per_tick: float   = 5.0
    half_spread_bps: float = 2.0
    depth_mean: float     = 1.0
    trade_prob: float     = 0.30
    trade_size_mean: float = 0.20
    tick_interval_us: float = 1_000.0
    seed: int | None      = None


def generate_ticks(config: SyntheticConfig) -> Iterator[MarketTick]:
    """
    Simple additive random walk. Not realistic, good enough to smoke-test strategies.
    Don't read too much into the PnL numbers from synthetic data.
    TODO: add informed flow and regime switching if you actually want calibration.
    """
    rng   = np.random.default_rng(config.seed)
    price = config.start_price
    ts    = 0.0

    for _ in range(config.n_ticks):
        price += float(rng.normal(0.0, config.vol_per_tick))
        price  = max(price, 1.0)

        half_spread = price * config.half_spread_bps / 10_000.0
        depth       = max(1e-6, float(rng.normal(config.depth_mean, config.depth_mean * 0.25)))

        book = OrderBook(
            bids=[BookLevel(price - half_spread, depth)],
            asks=[BookLevel(price + half_spread, depth)],
            ts=ts,
        )
        trades = _maybe_trade(rng, config, book, ts)
        yield MarketTick(book=book, trades=trades, ts=ts)
        ts += config.tick_interval_us


def _maybe_trade(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    book: OrderBook,
    ts: float,
) -> list[Trade]:
    if rng.random() >= cfg.trade_prob:
        return []
    side  = Side.BUY if rng.random() < 0.5 else Side.SELL
    price = book.asks[0].price if side == Side.BUY else book.bids[0].price
    return [Trade(price=price, size=float(rng.exponential(cfg.trade_size_mean)), side=side, ts=ts)]
