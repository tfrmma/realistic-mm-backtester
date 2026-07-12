from __future__ import annotations

import uuid

import pytest

from mmbt.core.types import Fill, Side
from mmbt.reporting.metrics import (
    BacktestReport,
    EquitySnapshot,
    FillRecord,
    StrategyMetrics,
    _max_drawdown,
    _sharpe,
)


def _snap(ts: float, realized: float, unrealized: float = 0.0, pos: float = 0.0) -> EquitySnapshot:
    return EquitySnapshot(ts=ts, realized_pnl=realized, unrealized_pnl=unrealized,
                          position=pos, fees_paid=0.0)


def _fill(side: Side, price: float, ts: float = 1000.0) -> Fill:
    return Fill(order_id=str(uuid.uuid4()), symbol="BTC-USD",
                side=side, price=price, size=0.1, is_maker=True, ts=ts)


class TestSharpe:
    def test_flat_equity_zero(self):
        assert _sharpe([100.0] * 20) == pytest.approx(0.0)

    def test_monotone_positive(self):
        assert _sharpe([float(i) for i in range(1, 21)]) > 0

    def test_single_snapshot(self):
        assert _sharpe([100.0]) == pytest.approx(0.0)

    def test_empty(self):
        assert _sharpe([]) == pytest.approx(0.0)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        assert _max_drawdown([0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_simple_drawdown(self):
        assert _max_drawdown([0.0, 10.0, 3.0, 8.0]) == pytest.approx(7.0)

    def test_empty(self):
        assert _max_drawdown([]) == pytest.approx(0.0)


class TestBacktestReport:
    def _metrics(self) -> StrategyMetrics:
        m = StrategyMetrics(symbol="BTC-USD")
        for i in range(10):
            m.equity_snapshots.append(_snap(float(i * 1000), float(i)))
            m.mid_history.append((float(i * 1000), 50_000.0 + i))
        return m

    def test_no_fills(self):
        report = BacktestReport.from_metrics(self._metrics())
        assert report.sharpe > 0
        assert report.adverse_scores == []

    def test_with_fill(self):
        m = self._metrics()
        f = _fill(Side.BUY, 50_000.0, ts=1000.0)
        m.fills.append(f)
        m.fill_records.append(FillRecord(fill=f, mid_at_fill=50_000.0))
        report = BacktestReport.from_metrics(m, lookback_ticks=3)
        assert isinstance(report.adverse_scores, list)

    def test_summary_keys(self):
        assert "n_fills" in BacktestReport.from_metrics(self._metrics()).metrics.summary()

    def test_print_no_crash(self, capsys):
        BacktestReport.from_metrics(self._metrics()).print_summary()
        assert "mmbt backtest" in capsys.readouterr().out


class TestMidHistoryCapacityInReport:
    def test_fill_within_retained_window_still_scored(self):
        m = StrategyMetrics(symbol="X", mid_history_capacity=15)
        f = _fill(Side.BUY, 100.0, ts=10_000.0)  # near the end -> still in the retained window
        m.fills.append(f)
        m.fill_records.append(FillRecord(fill=f, mid_at_fill=100.0))
        for i in range(20):  # exceeds capacity=15 -> the buffer wraps
            m.mid_history.append((float(i * 1000), 100.0))
            m.equity_snapshots.append(_snap(float(i * 1000), 0.0))
        report = BacktestReport.from_metrics(m, lookback_ticks=3)
        assert len(report.adverse_scores) == 1

    def test_fill_evicted_by_wraparound_is_silently_skipped_not_an_error(self):
        m = StrategyMetrics(symbol="X", mid_history_capacity=5)
        f = _fill(Side.BUY, 100.0, ts=0.0)  # tick 0 -- will be evicted once buffer wraps
        m.fills.append(f)
        m.fill_records.append(FillRecord(fill=f, mid_at_fill=100.0))
        for i in range(50):  # way past capacity=5, tick 0's mid point is long gone
            m.mid_history.append((float(i * 1000), 100.0))
            m.equity_snapshots.append(_snap(float(i * 1000), 0.0))
        report = BacktestReport.from_metrics(m, lookback_ticks=3)
        assert report.adverse_scores == []  # no crash, just nothing to score


class TestAdverseSelectionSign:
    def _report(self, side: Side, mid_at: float, mid_after: float) -> BacktestReport:
        m = StrategyMetrics(symbol="X")
        f = _fill(side, mid_at, ts=0.0)
        m.fills.append(f)
        m.fill_records.append(FillRecord(fill=f, mid_at_fill=mid_at))
        for i in range(20):
            m.mid_history.append((float(i * 1000), mid_at if i < 10 else mid_after))
            m.equity_snapshots.append(_snap(float(i * 1000), 0.0))
        return BacktestReport.from_metrics(m, lookback_ticks=10)

    def test_buy_adverse_positive(self):
        assert self._report(Side.BUY, 100.0, 90.0).adverse_selection_score > 0

    def test_buy_favorable_negative(self):
        assert self._report(Side.BUY, 100.0, 110.0).adverse_selection_score < 0

    def test_sell_adverse_positive(self):
        assert self._report(Side.SELL, 100.0, 110.0).adverse_selection_score > 0

    def test_sell_favorable_negative(self):
        assert self._report(Side.SELL, 100.0, 90.0).adverse_selection_score < 0
