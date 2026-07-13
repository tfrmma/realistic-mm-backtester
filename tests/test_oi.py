from __future__ import annotations

import uuid

import pytest

from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import BookLevel, MarketTick, OrderBook
from mmbt.data.open_interest import OpenInterestSchedule, generate_oi_schedule
from mmbt.engine.simple import BacktestEngine


class TestOpenInterestSchedule:
    def test_empty_schedule_returns_none(self):
        assert OpenInterestSchedule([]).as_of(1000.0) is None

    def test_sorts_unsorted_input(self):
        sched = OpenInterestSchedule([(2000.0, 20.0), (0.0, 10.0), (1000.0, 15.0)])
        assert sched.as_of(0.0) == pytest.approx(10.0)
        assert sched.as_of(1000.0) == pytest.approx(15.0)
        assert sched.as_of(2000.0) == pytest.approx(20.0)

    def test_forward_fill_between_points(self):
        sched = OpenInterestSchedule([(0.0, 10.0), (1000.0, 20.0)])
        assert sched.as_of(500.0) == pytest.approx(10.0)  # last known value, not interpolated

    def test_exact_match(self):
        sched = OpenInterestSchedule([(0.0, 10.0), (1000.0, 20.0)])
        assert sched.as_of(1000.0) == pytest.approx(20.0)

    def test_before_first_point_returns_none(self):
        sched = OpenInterestSchedule([(1000.0, 10.0)])
        assert sched.as_of(0.0) is None

    def test_after_last_point_forward_fills(self):
        sched = OpenInterestSchedule([(0.0, 10.0), (1000.0, 20.0)])
        assert sched.as_of(1_000_000.0) == pytest.approx(20.0)

    def test_len(self):
        assert len(OpenInterestSchedule([(0.0, 1.0), (1.0, 2.0)])) == 2

    def test_from_dict(self):
        sched = OpenInterestSchedule.from_dict({0.0: 10.0, 1000.0: 20.0})
        assert sched.as_of(1000.0) == pytest.approx(20.0)

    def test_from_csv(self, tmp_path):
        p = tmp_path / "oi.csv"
        p.write_text("ts,oi\n0.0,10000000.0\n60000000.0,10500000.0\n120000000.0,9800000.0\n")
        sched = OpenInterestSchedule.from_csv(p)
        assert sched.as_of(60000000.0) == pytest.approx(10500000.0)
        assert sched.as_of(90000000.0) == pytest.approx(10500000.0)  # forward-fill

    def test_from_csv_custom_columns(self, tmp_path):
        p = tmp_path / "oi.csv"
        p.write_text("timestamp,open_interest\n0.0,5.0\n1000.0,6.0\n")
        sched = OpenInterestSchedule.from_csv(p, ts_col="timestamp", oi_col="open_interest")
        assert sched.as_of(1000.0) == pytest.approx(6.0)

    def test_from_csv_missing_columns_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("ts,not_oi\n0.0,1.0\n")
        with pytest.raises(ValueError, match="missing columns"):
            OpenInterestSchedule.from_csv(p)


class TestChange:
    def test_positive_change(self):
        sched = OpenInterestSchedule([(0.0, 100.0), (1000.0, 150.0)])
        assert sched.change(1000.0, lookback_us=1000.0) == pytest.approx(50.0)

    def test_negative_change(self):
        sched = OpenInterestSchedule([(0.0, 100.0), (1000.0, 60.0)])
        assert sched.change(1000.0, lookback_us=1000.0) == pytest.approx(-40.0)

    def test_none_when_lookback_predates_schedule(self):
        sched = OpenInterestSchedule([(1000.0, 100.0), (2000.0, 150.0)])
        assert sched.change(2000.0, lookback_us=5000.0) is None

    def test_zero_change_when_no_update_in_window(self):
        sched = OpenInterestSchedule([(0.0, 100.0)])
        assert sched.change(1000.0, lookback_us=500.0) == pytest.approx(0.0)


class TestGenerateOiSchedule:
    def test_deterministic_with_seed(self):
        a = generate_oi_schedule(n_points=50, seed=42)
        b = generate_oi_schedule(n_points=50, seed=42)
        assert [a.as_of(float(i)) for i in range(0, 50 * 60_000_000, 60_000_000)] == \
               [b.as_of(float(i)) for i in range(0, 50 * 60_000_000, 60_000_000)]

    def test_different_seeds_differ(self):
        a = generate_oi_schedule(n_points=50, seed=1)
        b = generate_oi_schedule(n_points=50, seed=2)
        assert len(a) == len(b) == 50
        # at least one point should differ between the two runs
        vals_a = [a.as_of(float(i * 60_000_000)) for i in range(50)]
        vals_b = [b.as_of(float(i * 60_000_000)) for i in range(50)]
        assert vals_a != vals_b

    def test_never_negative(self):
        sched = generate_oi_schedule(n_points=200, start_oi=100.0, vol=1000.0, seed=0)
        for i in range(200):
            v = sched.as_of(float(i * 60_000_000))
            assert v is not None and v >= 0.0

    def test_correct_length_and_spacing(self):
        sched = generate_oi_schedule(n_points=10, interval_us=1000.0, seed=0)
        assert len(sched) == 10
        assert sched.as_of(9000.0) is not None
        assert sched.as_of(-1.0) is None  # nothing before the first point


class TestOiScheduleConsultedByStrategy:
    """Confirms the documented pattern actually works end-to-end: the engine
    knows nothing about OI, the strategy just holds a reference and queries
    book.ts itself no engine/protocol changes required."""

    def test_strategy_reads_oi_without_any_engine_support(self):
        oi_sched = OpenInterestSchedule([(0.0, 100.0), (2000.0, 200.0), (4000.0, 300.0)])

        class _OiAwareStrategy(BaseStrategy):
            def __init__(self, oi_sched: OpenInterestSchedule) -> None:
                self.oi_sched = oi_sched
                self.seen: list[float | None] = []

            def on_tick(self, book, trades):
                self.seen.append(self.oi_sched.as_of(book.ts))
                return []

        strat  = _OiAwareStrategy(oi_sched)
        engine = BacktestEngine()
        engine.add_strategy("mm", strat, "X")

        book  = OrderBook(bids=[BookLevel(99.0, 1.0)], asks=[BookLevel(101.0, 1.0)], ts=0.0)
        ticks = [MarketTick(book=book, trades=[], ts=float(i * 1000)) for i in range(6)]
        # book.ts must reflect each tick's own ts for the lookup to make sense
        ticks = [MarketTick(book=OrderBook(bids=book.bids, asks=book.asks, ts=t.ts), trades=[], ts=t.ts) for t in ticks]
        engine.run(ticks)

        assert strat.seen == [100.0, 100.0, 200.0, 200.0, 300.0, 300.0]
