from __future__ import annotations

import uuid

import pytest

from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import BookLevel, MarketTick, Order, OrderBook, Side
from mmbt.engine.pro import ProBacktestEngine
from mmbt.engine.simple import BacktestEngine
from mmbt.latency.config import LatencyConfig
from mmbt.queue.taker import crosses_book, sweep_book


def _book(bid: float = 100.0, ask: float = 101.0, bid_sz: float = 1.0, ask_sz: float = 1.0) -> OrderBook:
    return OrderBook(bids=[BookLevel(bid, bid_sz)], asks=[BookLevel(ask, ask_sz)], ts=0.0)


def _order(side: Side, price: float, size: float = 1.0, post_only: bool = False) -> Order:
    return Order(str(uuid.uuid4()), "X", side, price, size, is_post_only=post_only)


class TestCrossesBook:
    def test_buy_at_or_above_ask_crosses(self):
        book = _book(bid=100.0, ask=101.0)
        assert crosses_book(_order(Side.BUY, 101.0), book) is True
        assert crosses_book(_order(Side.BUY, 105.0), book) is True

    def test_buy_below_ask_does_not_cross(self):
        assert crosses_book(_order(Side.BUY, 100.5), _book(bid=100.0, ask=101.0)) is False

    def test_sell_at_or_below_bid_crosses(self):
        book = _book(bid=100.0, ask=101.0)
        assert crosses_book(_order(Side.SELL, 100.0), book) is True
        assert crosses_book(_order(Side.SELL, 95.0), book) is True

    def test_sell_above_bid_does_not_cross(self):
        assert crosses_book(_order(Side.SELL, 100.5), _book(bid=100.0, ask=101.0)) is False

    def test_empty_book_never_crosses(self):
        empty = OrderBook(bids=[], asks=[], ts=0.0)
        assert crosses_book(_order(Side.BUY, 1_000_000.0), empty) is False


class TestSweepBook:
    def test_single_level_full_fill(self):
        execution = sweep_book(_order(Side.BUY, 101.0, size=2.0), _book(ask=101.0, ask_sz=5.0))
        assert execution is not None
        assert execution.filled_size == pytest.approx(2.0)
        assert execution.vwap_price == pytest.approx(101.0)
        assert execution.levels_consumed == 1

    def test_size_capped_by_available_depth(self):
        # book only has 1.5 a taker order never invents liquidity beyond that
        execution = sweep_book(_order(Side.BUY, 101.0, size=5.0), _book(ask=101.0, ask_sz=1.5))
        assert execution is not None
        assert execution.filled_size == pytest.approx(1.5)

    def test_multi_level_vwap(self):
        book = OrderBook(
            bids=[BookLevel(99.0, 10.0)],
            asks=[BookLevel(101.0, 1.0), BookLevel(102.0, 1.0)],
            ts=0.0,
        )
        execution = sweep_book(_order(Side.BUY, 102.0, size=2.0), book)
        assert execution is not None
        assert execution.filled_size == pytest.approx(2.0)
        assert execution.vwap_price == pytest.approx((101.0 + 102.0) / 2.0)
        assert execution.levels_consumed == 2

    def test_limit_price_stops_the_sweep(self):
        book = OrderBook(
            bids=[BookLevel(99.0, 10.0)],
            asks=[BookLevel(101.0, 1.0), BookLevel(105.0, 10.0)],
            ts=0.0,
        )
        # limit only accepts up to 101.0 must never touch the 105.0 level
        execution = sweep_book(_order(Side.BUY, 101.0, size=5.0), book)
        assert execution is not None
        assert execution.filled_size == pytest.approx(1.0)
        assert execution.levels_consumed == 1

    def test_non_crossing_order_returns_none(self):
        assert sweep_book(_order(Side.BUY, 100.0), _book(bid=100.0, ask=101.0)) is None

    def test_sell_sweep(self):
        execution = sweep_book(_order(Side.SELL, 100.0, size=2.0), _book(bid=100.0, bid_sz=3.0))
        assert execution is not None
        assert execution.filled_size == pytest.approx(2.0)
        assert execution.vwap_price == pytest.approx(100.0)


class _TakerOnce(BaseStrategy):
    def __init__(self, side: Side, price: float, size: float = 1.0, post_only: bool = False) -> None:
        self.side, self.price, self.size, self.post_only = side, price, size, post_only
        self._sent = False

    def on_tick(self, book, trades):
        if self._sent:
            return []
        self._sent = True
        return [Order(str(uuid.uuid4()), "X", self.side, self.price, self.size, is_post_only=self.post_only)]


class TestTakerInSimpleEngine:
    def test_crossing_order_fills_immediately_as_taker(self):
        engine = BacktestEngine(fee_rate_taker=0.001)
        engine.add_strategy("mm", _TakerOnce(Side.BUY, 101.0), "X")
        ticks = [MarketTick(book=_book(ask_sz=5.0), trades=[], ts=0.0)]
        m = engine.run(ticks)["mm"]
        assert len(m.fills) == 1
        assert m.fills[0].is_maker is False
        assert m.fills[0].price == pytest.approx(101.0)
        assert m.fees_paid > 0.0
        assert engine._strats["mm"].pending == []

    def test_post_only_crossing_order_is_rejected(self):
        engine = BacktestEngine()
        engine.add_strategy("mm", _TakerOnce(Side.BUY, 101.0, post_only=True), "X")
        ticks = [MarketTick(book=_book(ask_sz=5.0), trades=[], ts=0.0)]
        m = engine.run(ticks)["mm"]
        assert len(m.fills) == 0
        assert m.rejected_orders == 1
        assert engine._strats["mm"].pending == []

    def test_non_crossing_post_only_still_rests_normally(self):
        engine = BacktestEngine()
        engine.add_strategy("mm", _TakerOnce(Side.BUY, 99.0, post_only=True), "X")
        engine.run([MarketTick(book=_book(), trades=[], ts=0.0)])
        assert len(engine._strats["mm"].pending) == 1
        assert engine._strats["mm"].metrics.rejected_orders == 0


class TestTakerInProEngine:
    def _engine(self, **kw) -> ProBacktestEngine:
        return ProBacktestEngine(
            latency_config=LatencyConfig(feed_us=0.01, order_us=500.0, cancel_us=300.0, jitter=0.05),
            seed=1, **kw,
        )

    def test_crossing_order_fills_as_taker_at_landing(self):
        book  = _book(ask_sz=5.0)
        ticks = [MarketTick(book=book, trades=[], ts=float(i * 1000)) for i in range(10)]
        engine = self._engine(fee_rate_taker=0.001)
        engine.add_strategy("mm", _TakerOnce(Side.BUY, 101.0), "X")
        m = engine.run(ticks)["mm"]
        assert len(m.fills) == 1
        assert m.fills[0].is_maker is False
        assert m.fees_paid > 0.0

    def test_post_only_crossing_order_rejected_at_landing(self):
        book  = _book(ask_sz=5.0)
        ticks = [MarketTick(book=book, trades=[], ts=float(i * 1000)) for i in range(10)]
        engine = self._engine()
        engine.add_strategy("mm", _TakerOnce(Side.BUY, 101.0, post_only=True), "X")
        m = engine.run(ticks)["mm"]
        assert len(m.fills) == 0
        assert m.rejected_orders == 1
