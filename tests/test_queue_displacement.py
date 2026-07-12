from __future__ import annotations

import uuid

import pytest

from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import BookLevel, CancelOrder, MarketTick, Order, OrderBook, Side, Trade
from mmbt.engine.pro import ProBacktestEngine
from mmbt.latency.config import LatencyConfig


def _book(bid: float = 100.0, ask: float = 101.0, depth: float = 5.0) -> OrderBook:
    return OrderBook(bids=[BookLevel(bid, depth)], asks=[BookLevel(ask, depth)], ts=0.0)


class _PostOnce(BaseStrategy):
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._sent  = False

    def on_tick(self, book, trades):
        if self._sent:
            return []
        self._sent = True
        return [Order(str(uuid.uuid4()), self.symbol, Side.BUY, 100.0, 1.0, is_post_only=True)]


class _OrderThenCancel(BaseStrategy):
    """Posts a resting BUY on its first on_tick call, cancels it on the
    second. Pure tick-counting, doesn't look at the book at all keeps the
    timeline fully deterministic."""
    def __init__(self, symbol: str, price: float = 100.0) -> None:
        self.symbol    = symbol
        self.price     = price
        self.order_id: str | None = None
        self._tick     = 0

    def on_tick(self, book, trades):
        self._tick += 1
        if self._tick == 1:
            self.order_id = str(uuid.uuid4())
            return [Order(self.order_id, self.symbol, Side.BUY, self.price, 1.0, is_post_only=True)]
        if self._tick == 2:
            return [CancelOrder(self.order_id, self.symbol)]
        return []


def _engine(cancel_us: float, order_us: float = 1000.0) -> ProBacktestEngine:
    # jitter=0.0 -> arrival = submit_ts + median exactly, fully deterministic timeline
    return ProBacktestEngine(
        latency_config=LatencyConfig(feed_us=0.01, order_us=order_us, cancel_us=cancel_us, jitter=0.0),
        seed=0,
    )


class TestQueueDisplacement:
    def test_no_cancel_no_displacement(self):
        book  = _book()
        ticks = [
            MarketTick(book=book, trades=[Trade(price=100.0, size=5.0, side=Side.SELL, ts=float(i * 1000))], ts=float(i * 1000))
            for i in range(10)
        ]
        engine = _engine(cancel_us=1000.0)
        engine.add_strategy("mm", _PostOnce("X"), "X")
        m = engine.run(ticks)["mm"]
        assert len(m.fills) >= 1
        assert all(f.queue_displacement_us == 0.0 for f in m.fills)

    def test_cancel_wins_race_no_fill(self):
        # matching trade only shows up long after the cancel has landed
        # cancel wins clean, order never fills
        book  = _book()
        ticks = []
        for i in range(6):
            ts = float(i * 1000)
            trades = [Trade(price=100.0, size=5.0, side=Side.SELL, ts=ts)] if i >= 5 else []
            ticks.append(MarketTick(book=book, trades=trades, ts=ts))
        engine = _engine(cancel_us=1000.0, order_us=1000.0)
        engine.add_strategy("mm", _OrderThenCancel("X"), "X")
        m = engine.run(ticks)["mm"]
        assert len(m.fills) == 0

    def test_fill_beats_late_cancel_displacement_is_positive(self):
        # order lands ts=1000 (order_us=1000). cancel submitted ts=1000,
        # cancel_us=2500 -> arrives ts=3500. A matching trade at ts=2000 fills
        # the order while the cancel is still in flight:
        # queue_displacement_us = 3500 - 1000 = 2500
        book  = _book()
        ticks = []
        for i in range(6):
            ts = float(i * 1000)
            trades = [Trade(price=100.0, size=5.0, side=Side.SELL, ts=ts)] if ts == 2000.0 else []
            ticks.append(MarketTick(book=book, trades=trades, ts=ts))
        engine = _engine(cancel_us=2500.0, order_us=1000.0)
        engine.add_strategy("mm", _OrderThenCancel("X"), "X")
        m = engine.run(ticks)["mm"]
        assert len(m.fills) == 1
        assert m.fills[0].queue_displacement_us == pytest.approx(2500.0)

    def test_no_dangling_state_after_race(self):
        book  = _book()
        ticks = []
        for i in range(6):
            ts = float(i * 1000)
            trades = [Trade(price=100.0, size=5.0, side=Side.SELL, ts=ts)] if ts == 2000.0 else []
            ticks.append(MarketTick(book=book, trades=trades, ts=ts))
        engine = _engine(cancel_us=2500.0, order_us=1000.0)
        engine.add_strategy("mm", _OrderThenCancel("X"), "X")
        engine.run(ticks)
        state = engine._strats["mm"]
        assert state.pending_cancels == {}
        assert state.order_register_ts == {}

    def test_no_dangling_state_after_clean_cancel(self):
        book  = _book()
        ticks = []
        for i in range(6):
            ts = float(i * 1000)
            trades = [Trade(price=100.0, size=5.0, side=Side.SELL, ts=ts)] if i >= 5 else []
            ticks.append(MarketTick(book=book, trades=trades, ts=ts))
        engine = _engine(cancel_us=1000.0, order_us=1000.0)
        engine.add_strategy("mm", _OrderThenCancel("X"), "X")
        engine.run(ticks)
        state = engine._strats["mm"]
        assert state.pending_cancels == {}
        assert state.order_register_ts == {}
