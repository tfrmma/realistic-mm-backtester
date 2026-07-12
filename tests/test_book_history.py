from __future__ import annotations

import pytest

from mmbt.core.types import BookLevel, OrderBook
from mmbt.latency.book_history import BookHistory


def _book(ts: float, price: float = 100.0) -> OrderBook:
    return OrderBook(bids=[BookLevel(price - 1, 1.0)], asks=[BookLevel(price + 1, 1.0)], ts=ts)


class TestBookHistory:
    def test_empty_returns_none(self):
        assert BookHistory().as_of(0.0) is None

    def test_invalid_maxlen_raises(self):
        with pytest.raises(ValueError):
            BookHistory(maxlen=0)

    def test_exact_match(self):
        h = BookHistory()
        h.push(1000.0, _book(1000.0, 100.0), [])
        h.push(2000.0, _book(2000.0, 200.0), [])
        snap = h.as_of(2000.0)
        assert snap is not None
        assert snap.ts == 2000.0
        assert snap.book.mid == pytest.approx(200.0)

    def test_returns_most_recent_leq_target(self):
        h = BookHistory()
        for ts in (0.0, 1000.0, 2000.0, 3000.0):
            h.push(ts, _book(ts, ts), [])
        snap = h.as_of(2500.0)
        assert snap is not None
        assert snap.ts == 2000.0

    def test_target_before_all_history_returns_oldest(self):
        h = BookHistory()
        h.push(5000.0, _book(5000.0, 500.0), [])
        h.push(6000.0, _book(6000.0, 600.0), [])
        snap = h.as_of(0.0)
        assert snap is not None
        assert snap.ts == 5000.0

    def test_maxlen_evicts_oldest(self):
        h = BookHistory(maxlen=3)
        for ts in (0.0, 1000.0, 2000.0, 3000.0):
            h.push(ts, _book(ts, ts), [])
        assert len(h) == 3
        # ts=0.0 was evicted; the oldest retained snapshot is now ts=1000.0
        snap = h.as_of(0.0)
        assert snap is not None
        assert snap.ts == 1000.0

    def test_len_tracks_pushes(self):
        h = BookHistory(maxlen=10)
        assert len(h) == 0
        for i in range(5):
            h.push(float(i), _book(float(i)), [])
        assert len(h) == 5

    def test_trades_carried_with_snapshot(self):
        from mmbt.core.types import Side, Trade
        h = BookHistory()
        trades = [Trade(price=100.0, size=1.0, side=Side.BUY, ts=1000.0)]
        h.push(1000.0, _book(1000.0), trades)
        snap = h.as_of(1000.0)
        assert snap is not None
        assert snap.trades == trades
