from __future__ import annotations

import uuid

import pytest

from mmbt.core.types import BookLevel, Order, OrderBook, Side, Trade
from mmbt.queue.cancel_models import ReduceRatioCancelModel
from mmbt.queue.fifo import FIFOQueueSimulator, _qty_in_front


def _book(bid: float, ask: float, depth: float = 1.0) -> OrderBook:
    return OrderBook(bids=[BookLevel(bid, depth)], asks=[BookLevel(ask, depth)], ts=0.0)


def _order(side: Side, price: float, size: float = 1.0) -> Order:
    return Order(order_id=str(uuid.uuid4()), symbol="BTC-USD", side=side, price=price, size=size)


def _trade(side: Side, price: float, size: float, ts: float = 1_000.0) -> Trade:
    return Trade(price=price, size=size, side=side, ts=ts)


def _sim(cancel_ratio: float = 0.0) -> FIFOQueueSimulator:
    return FIFOQueueSimulator(ReduceRatioCancelModel(cancel_ratio), use_rust=False)


class TestQtyInFront:
    def test_buy_at_best_all_depth_in_front(self):
        book  = _book(100.0, 101.0, depth=5.0)
        order = _order(Side.BUY, 100.0)
        assert _qty_in_front(order, book) == pytest.approx(5.0)

    def test_buy_with_better_levels(self):
        book = OrderBook(
            bids=[BookLevel(101.0, 3.0), BookLevel(100.0, 5.0)],
            asks=[BookLevel(102.0, 2.0)], ts=0.0,
        )
        assert _qty_in_front(_order(Side.BUY, 100.0), book) == pytest.approx(8.0)

    def test_new_tighter_level_no_queue(self):
        book  = _book(99.0, 101.0, depth=5.0)
        assert _qty_in_front(_order(Side.BUY, 100.0), book) == pytest.approx(0.0)

    def test_sell_side(self):
        book = OrderBook(
            bids=[BookLevel(99.0, 2.0)],
            asks=[BookLevel(100.0, 4.0), BookLevel(101.0, 3.0)], ts=0.0,
        )
        assert _qty_in_front(_order(Side.SELL, 100.0), book) == pytest.approx(4.0)


class TestFIFOFills:
    def test_fill_at_front_of_empty_level(self):
        sim  = _sim()
        book = OrderBook(bids=[BookLevel(100.0, 0.0)], asks=[BookLevel(101.0, 2.0)], ts=0.0)
        sim.register(_order(Side.BUY, 100.0, size=1.0), book)
        fills = sim.process_tick(book, [_trade(Side.SELL, 100.0, 2.0)])
        assert len(fills) == 1
        assert fills[0].size == pytest.approx(1.0)
        assert fills[0].is_maker is True

    def test_no_fill_queue_not_exhausted(self):
        sim  = _sim()
        book = _book(100.0, 101.0, depth=10.0)
        sim.register(_order(Side.BUY, 100.0), book)
        assert sim.process_tick(book, [_trade(Side.SELL, 100.0, 2.0)]) == []

    def test_fill_with_cancel_model(self):
        # cancel_ratio=0.5 -> trade of 8 consumes 12 -> clears 10 in front with 2 overshoot
        sim  = _sim(cancel_ratio=0.5)
        book = _book(100.0, 101.0, depth=10.0)
        sim.register(_order(Side.BUY, 100.0, size=1.0), book)
        fills = sim.process_tick(book, [_trade(Side.SELL, 100.0, 8.0)])
        assert len(fills) == 1

    def test_wrong_side_no_fill(self):
        sim  = _sim()
        book = _book(100.0, 101.0, depth=0.0)
        sim.register(_order(Side.BUY, 100.0), book)
        assert sim.process_tick(book, [_trade(Side.BUY, 100.0, 5.0)]) == []


class TestCancel:
    def test_cancel_active_order(self):
        sim  = _sim()
        book = _book(100.0, 101.0)
        o    = _order(Side.BUY, 100.0)
        sim.register(o, book)
        assert sim.pending_count() == 1
        assert sim.cancel(o.order_id) is True
        assert sim.pending_count() == 0

    def test_cancel_nonexistent(self):
        assert _sim().cancel("ghost") is False

    def test_cancelled_order_not_filled(self):
        sim  = _sim()
        book = OrderBook(bids=[BookLevel(100.0, 0.0)], asks=[BookLevel(101.0, 1.0)], ts=0.0)
        o    = _order(Side.BUY, 100.0)
        sim.register(o, book)
        sim.cancel(o.order_id)
        assert sim.process_tick(book, [_trade(Side.SELL, 100.0, 5.0)]) == []

    def test_double_cancel_returns_false(self):
        sim  = _sim()
        book = _book(100.0, 101.0)
        o    = _order(Side.BUY, 100.0)
        sim.register(o, book)
        sim.cancel(o.order_id)
        assert sim.cancel(o.order_id) is False
