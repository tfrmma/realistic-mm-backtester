from __future__ import annotations

import uuid

import pytest

from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import BookLevel, MarketTick, Order, OrderBook, Side, Trade
from mmbt.engine.simple import BacktestEngine
from mmbt.queue.passive import PassiveFillSimulator


class _OneShotOrder(BaseStrategy):
    """Posts a single resting BUY once, never cancels, never re-quotes."""

    def __init__(self, symbol: str, price: float, size: float) -> None:
        self.symbol   = symbol
        self.price    = price
        self.size     = size
        self._posted  = False

    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list:
        if self._posted:
            return []
        self._posted = True
        return [Order(str(uuid.uuid4()), self.symbol, Side.BUY, self.price, self.size, is_post_only=True)]


def _book(price: float = 100.0, depth: float = 2.0) -> OrderBook:
    return OrderBook(bids=[BookLevel(price, depth)], asks=[BookLevel(price + 1.0, depth)], ts=0.0)


def _crossing_trade(price: float, size: float, ts: float) -> Trade:
    return Trade(price=price, size=size, side=Side.SELL, ts=ts)


class TestPassiveFillSimulatorPartial:
    def test_fill_smaller_than_order_when_depth_thin(self):
        sim   = PassiveFillSimulator(fill_ratio=0.5)
        order = Order("o1", "X", Side.BUY, 100.0, 1.0)
        book  = _book(price=100.0, depth=2.0)
        fill  = sim.simulate(order, [_crossing_trade(100.0, 1.0, 0.0)], book)
        assert fill is not None
        assert 0.0 < fill.size < order.size

    def test_full_fill_when_traded_volume_covers_size(self):
        sim   = PassiveFillSimulator(fill_ratio=1.0)
        order = Order("o1", "X", Side.BUY, 100.0, 1.0)
        book  = _book(price=100.0, depth=1.0)
        fill  = sim.simulate(order, [_crossing_trade(100.0, 5.0, 0.0)], book)
        assert fill is not None
        assert fill.size == pytest.approx(order.size)


class TestEngineKeepsPartialOrderResting:
    def test_partial_fills_accumulate_to_full_size(self):
        engine = BacktestEngine(fill_sim=PassiveFillSimulator(fill_ratio=0.5), snapshot_every=1000)
        engine.add_strategy("mm", _OneShotOrder("X", price=100.0, size=1.0), "X")

        book  = _book(price=100.0, depth=2.0)
        ticks = [
            MarketTick(book=book, trades=[_crossing_trade(100.0, 1.0, float(i))], ts=float(i))
            for i in range(200)
        ]
        m = engine.run(ticks)["mm"]

        assert len(m.fills) > 1  # took more than one tick to fill -- proves it's partial, not one-shot
        total_filled = sum(f.size for f in m.fills)
        assert total_filled == pytest.approx(1.0, abs=1e-6)
        assert engine._strats["mm"].pending == []  # fully consumed, nothing left resting

    def test_no_overfill_even_with_excess_ticks(self):
        engine = BacktestEngine(fill_sim=PassiveFillSimulator(fill_ratio=0.5), snapshot_every=1000)
        engine.add_strategy("mm", _OneShotOrder("X", price=100.0, size=1.0), "X")

        book  = _book(price=100.0, depth=2.0)
        ticks = [
            MarketTick(book=book, trades=[_crossing_trade(100.0, 1.0, float(i))], ts=float(i))
            for i in range(500)  # far beyond the ~100 ticks needed to fully converge
        ]
        m = engine.run(ticks)["mm"]
        total_filled = sum(f.size for f in m.fills)
        assert total_filled <= 1.0 + 1e-9
        assert engine._strats["mm"].pending == []

    def test_single_tick_full_fill_still_removes_order(self):
        engine = BacktestEngine(fill_sim=PassiveFillSimulator(fill_ratio=1.0), snapshot_every=1000)
        engine.add_strategy("mm", _OneShotOrder("X", price=100.0, size=1.0), "X")

        book  = _book(price=100.0, depth=1.0)
        ticks = [
            MarketTick(book=book, trades=[], ts=0.0),  # tick 0: order gets posted, nothing to fill yet
            MarketTick(book=book, trades=[_crossing_trade(100.0, 5.0, 1000.0)], ts=1000.0),  # tick 1: fills fully
        ]
        m = engine.run(ticks)["mm"]
        assert len(m.fills) == 1
        assert m.fills[0].size == pytest.approx(1.0)
        assert engine._strats["mm"].pending == []

    def test_cancel_removes_partially_filled_order(self):
        # a partially-filled order must still be cancellable by order_id like any other
        from mmbt.core.types import CancelOrder

        class _CancelAfterOneTick(BaseStrategy):
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol
                self._order_id: str | None = None
                self._tick = 0

            def on_tick(self, book: OrderBook, trades: list[Trade]) -> list:
                self._tick += 1
                if self._tick == 1:
                    self._order_id = str(uuid.uuid4())
                    return [Order(self._order_id, self.symbol, Side.BUY, 100.0, 1.0, is_post_only=True)]
                if self._tick == 2:
                    return [CancelOrder(self._order_id, self.symbol)]
                return []

        engine = BacktestEngine(fill_sim=PassiveFillSimulator(fill_ratio=0.5), snapshot_every=1000)
        engine.add_strategy("mm", _CancelAfterOneTick("X"), "X")
        book  = _book(price=100.0, depth=2.0)
        ticks = [
            MarketTick(book=book, trades=[_crossing_trade(100.0, 1.0, float(i))], ts=float(i))
            for i in range(5)
        ]
        m = engine.run(ticks)["mm"]
        assert len(m.fills) == 1          # only the tick-1 partial fill landed before the cancel
        assert engine._strats["mm"].pending == []
