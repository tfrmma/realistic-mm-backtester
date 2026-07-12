from __future__ import annotations

import uuid

import pytest

from mmbt.core.protocol import BaseMultiAssetStrategy
from mmbt.core.types import BookLevel, MarketTick, Order, OrderBook, Side
from mmbt.engine.multi_asset import MultiAssetEngine, _merge_by_ts


def _tick(ts: float, price: float = 100.0) -> MarketTick:
    book = OrderBook(bids=[BookLevel(price - 1.0, 5.0)], asks=[BookLevel(price + 1.0, 5.0)], ts=ts)
    return MarketTick(book=book, trades=[], ts=ts)


class TestMergeByTs:
    def test_interleaves_chronologically(self):
        streams = {
            "A": [_tick(0.0), _tick(2000.0), _tick(4000.0)],
            "B": [_tick(1000.0), _tick(3000.0)],
        }
        merged = list(_merge_by_ts(streams))
        assert [tick.ts for _, tick in merged] == [0.0, 1000.0, 2000.0, 3000.0, 4000.0]
        assert [sym for sym, _ in merged] == ["A", "B", "A", "B", "A"]

    def test_handles_uneven_stream_lengths(self):
        streams = {"A": [_tick(0.0)], "B": [_tick(1.0), _tick(2.0), _tick(3.0)]}
        assert len(list(_merge_by_ts(streams))) == 4

    def test_empty_stream_ignored(self):
        assert len(list(_merge_by_ts({"A": [_tick(0.0)], "B": []}))) == 1

    def test_all_empty(self):
        assert list(_merge_by_ts({"A": [], "B": []})) == []

    def test_single_stream(self):
        merged = list(_merge_by_ts({"A": [_tick(0.0), _tick(1000.0)]}))
        assert [sym for sym, _ in merged] == ["A", "A"]


class _RecordingStrategy(BaseMultiAssetStrategy):
    """Posts one resting BUY per symbol the first time it sees that symbol,
    records every (symbol, ts) it's called with."""
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self._posted: set[str] = set()

    def on_tick(self, symbol, book, trades):
        self.calls.append((symbol, book.ts))
        if symbol in self._posted:
            return []
        self._posted.add(symbol)
        return [Order(str(uuid.uuid4()), symbol, Side.BUY, book.mid - 1.0, 0.1, is_post_only=True)]


class TestMultiAssetEngine:
    def test_strategy_sees_both_symbols_with_correct_routing(self):
        strat  = _RecordingStrategy()
        engine = MultiAssetEngine(strat)
        streams = {
            "BTC-USD": [_tick(0.0, 100.0), _tick(2000.0, 101.0)],
            "ETH-USD": [_tick(1000.0, 50.0), _tick(3000.0, 51.0)],
        }
        results = engine.run(streams)
        assert set(results.keys()) == {"BTC-USD", "ETH-USD"}
        assert strat.calls == [
            ("BTC-USD", 0.0), ("ETH-USD", 1000.0), ("BTC-USD", 2000.0), ("ETH-USD", 3000.0),
        ]

    def test_each_symbol_gets_its_own_metrics(self):
        strat  = _RecordingStrategy()
        engine = MultiAssetEngine(strat)
        streams = {
            "A": [_tick(0.0, 100.0), _tick(1000.0, 100.0)],
            "B": [_tick(500.0, 200.0), _tick(1500.0, 200.0)],
        }
        results = engine.run(streams)
        assert results["A"].symbol == "A"
        assert results["B"].symbol == "B"
        # a fill on one symbol never lands in the other's fill list
        assert all(f.symbol == "A" for f in results["A"].fills)
        assert all(f.symbol == "B" for f in results["B"].fills)

    def test_taker_order_fills_immediately(self):
        class _Taker(BaseMultiAssetStrategy):
            def __init__(self) -> None:
                self._sent = False

            def on_tick(self, symbol, book, trades):
                if self._sent:
                    return []
                self._sent = True
                return [Order(str(uuid.uuid4()), symbol, Side.BUY, book.asks[0].price, 1.0, is_post_only=False)]

        engine = MultiAssetEngine(_Taker(), fee_rate_taker=0.001)
        results = engine.run({"X": [_tick(0.0, 100.0)]})
        m = results["X"]
        assert len(m.fills) == 1
        assert m.fills[0].is_maker is False
        assert m.fees_paid > 0.0

    def test_post_only_crossing_order_rejected(self):
        class _BadPostOnly(BaseMultiAssetStrategy):
            def __init__(self) -> None:
                self._sent = False

            def on_tick(self, symbol, book, trades):
                if self._sent:
                    return []
                self._sent = True
                return [Order(str(uuid.uuid4()), symbol, Side.BUY, book.asks[0].price, 1.0, is_post_only=True)]

        engine  = MultiAssetEngine(_BadPostOnly())
        results = engine.run({"X": [_tick(0.0, 100.0)]})
        assert results["X"].rejected_orders == 1
        assert len(results["X"].fills) == 0

    def test_snapshot_every_is_per_symbol_not_global(self):
        # regression: snapshot_every used to be gated on a counter shared
        # across ALL symbols, not per-symbol. With two streams ticking in
        # perfect lockstep that deterministically starved one symbol of
        # every single snapshot (always landed on the other) instead of
        # splitting them evenly.
        strat  = _RecordingStrategy()
        engine = MultiAssetEngine(strat, snapshot_every=2)
        streams = {
            "A": [_tick(float(i * 1000)) for i in range(20)],
            "B": [_tick(float(i * 1000)) for i in range(20)],
        }
        results = engine.run(streams)
        assert len(results["A"].equity_snapshots) == 10
        assert len(results["B"].equity_snapshots) == 10

    def test_cancel_removes_pending_order(self):
        from mmbt.core.types import CancelOrder

        class _PostThenCancel(BaseMultiAssetStrategy):
            def __init__(self) -> None:
                self.order_id: str | None = None
                self._tick = 0

            def on_tick(self, symbol, book, trades):
                self._tick += 1
                if self._tick == 1:
                    self.order_id = str(uuid.uuid4())
                    return [Order(self.order_id, symbol, Side.BUY, book.mid - 1.0, 0.1, is_post_only=True)]
                if self._tick == 2:
                    return [CancelOrder(self.order_id, symbol)]
                return []

        engine = MultiAssetEngine(_PostThenCancel())
        engine.run({"X": [_tick(0.0), _tick(1000.0), _tick(2000.0)]})
        assert engine._symbols["X"].pending == []
