from __future__ import annotations

import uuid

import pytest

from mmbt.core.types import BookLevel, MarketTick, Order, OrderBook, Side
from mmbt.data import SyntheticConfig, TickLoader
from mmbt.queue.cancel_models import CancelModel, ProbQueueCancelModel, ReduceRatioCancelModel
from mmbt.queue.fifo import RUST_AVAILABLE, FIFOQueueSimulator, _build_rust_core

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not compiled")


class _CustomCancelModel:
    """Arbitrary user CancelModel Rust can't represent -- always cancels everything."""
    def cancelled_fraction(self, qty_in_front: float, trade_size: float) -> float:
        return 1.0


def _order(side: Side, price: float, size: float = 0.1) -> Order:
    return Order(order_id=str(uuid.uuid4()), symbol="X", side=side, price=price, size=size, ts=0.0)


def _run_both(cancel_model: CancelModel, ticks: list[MarketTick], n_active: int = 20):
    """Replay the same order-registration schedule against Python and Rust
    paths and return (python_fills, rust_fills). Registers the identical
    Order object (same order_id) into both so fills are directly comparable —
    neither path mutates the shared object, so sharing it is safe."""
    py  = FIFOQueueSimulator(cancel_model, use_rust=False)
    rs  = FIFOQueueSimulator(cancel_model, use_rust=True)
    assert rs.using_rust is True
    assert py.using_rust is False

    def _register_pair(book) -> None:
        buy, sell = _order(Side.BUY, book.best_bid.price), _order(Side.SELL, book.best_ask.price)
        py.register(buy, book);  rs.register(buy, book)
        py.register(sell, book); rs.register(sell, book)

    first = ticks[0].book
    for _ in range(n_active // 2):
        _register_pair(first)

    py_fills, rs_fills = [], []
    for tick in ticks:
        book = tick.book
        py_fills.extend(py.process_tick(book, tick.trades))
        rs_fills.extend(rs.process_tick(book, tick.trades))
        needed = n_active - py.pending_count()
        for _ in range(max(0, needed // 2)):
            _register_pair(book)
    return py_fills, rs_fills


def _fill_key(f):
    return (f.order_id, round(f.price, 9), round(f.size, 9), f.is_maker, round(f.ts, 6))


class TestRustPythonParity:
    def test_reduce_ratio_model_identical_fills(self):
        ticks = TickLoader.synthetic(SyntheticConfig(
            n_ticks=3_000, vol_per_tick=5.0, trade_prob=0.35, seed=1,
        )).to_list()
        py_fills, rs_fills = _run_both(ReduceRatioCancelModel(0.20), ticks)
        assert len(py_fills) > 0  # sanity: the scenario actually produces fills
        assert len(py_fills) == len(rs_fills)
        assert sorted(map(_fill_key, py_fills)) == sorted(map(_fill_key, rs_fills))

    def test_prob_queue_model_identical_fills(self):
        ticks = TickLoader.synthetic(SyntheticConfig(
            n_ticks=3_000, vol_per_tick=5.0, trade_prob=0.35, seed=2,
        )).to_list()
        py_fills, rs_fills = _run_both(ProbQueueCancelModel(min_ratio=0.05, max_ratio=0.70), ticks)
        assert len(py_fills) > 0
        assert len(py_fills) == len(rs_fills)
        assert sorted(map(_fill_key, py_fills)) == sorted(map(_fill_key, rs_fills))

    def test_prob_queue_and_reduce_ratio_produce_different_dynamics(self):
        # deep book relative to trade size: ReduceRatio(0.20) depletes the
        # queue at a constant 20%-of-trade-size the whole way. ProbQueue(0.05,
        # 0.70) starts pinned at its 5% floor while trade_size/qty_in_front is
        # tiny, only rising as the queue thins -- consumption rates differ
        # from the very first trade, so ticks-to-first-fill must differ too.
        # (Guards against the old bug where ProbQueue silently got routed to
        # a fixed 0.20 ratio in Rust -- that would make these two match.)
        from mmbt.core.types import OrderBook, Side, Trade

        def _ticks_to_first_fill(cancel_model: CancelModel) -> int | None:
            sim   = FIFOQueueSimulator(cancel_model, use_rust=True)
            book0 = OrderBook(bids=[BookLevel(100.0, 1_000.0)], asks=[BookLevel(101.0, 1_000.0)], ts=0.0)
            sim.register(_order(Side.BUY, 100.0, size=1.0), book0)
            for i in range(500):
                trade = Trade(price=100.0, size=5.0, side=Side.SELL, ts=float(i))
                if sim.process_tick(book0, [trade]):
                    return i
            return None

        n_reduce = _ticks_to_first_fill(ReduceRatioCancelModel(0.20))
        n_prob   = _ticks_to_first_fill(ProbQueueCancelModel(min_ratio=0.05, max_ratio=0.70))
        assert n_reduce is not None and n_prob is not None
        assert n_reduce != n_prob


class TestRustCoreSelection:
    def test_reduce_ratio_maps_to_rust(self):
        sim = FIFOQueueSimulator(ReduceRatioCancelModel(0.3), use_rust=True)
        assert sim.using_rust is True

    def test_prob_queue_maps_to_rust(self):
        sim = FIFOQueueSimulator(ProbQueueCancelModel(0.1, 0.6), use_rust=True)
        assert sim.using_rust is True

    def test_custom_model_falls_back_to_python(self):
        sim = FIFOQueueSimulator(_CustomCancelModel(), use_rust=True)
        assert sim.using_rust is False  # Rust can't represent an arbitrary Python model

    def test_build_rust_core_returns_none_for_unknown_model(self):
        assert _build_rust_core(_CustomCancelModel()) is None

    def test_use_rust_false_always_uses_python(self):
        sim = FIFOQueueSimulator(ReduceRatioCancelModel(0.2), use_rust=False)
        assert sim.using_rust is False
