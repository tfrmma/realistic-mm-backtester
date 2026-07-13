from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from mmbt.core.types import BookLevel, MarketTick, OrderBook
from mmbt.engine.sweep import ParameterSweep, SweepResult, expand_grid
from mmbt.reporting.metrics import BacktestReport, EquitySnapshot, StrategyMetrics
from mmbt.reporting.sweep_plots import equity_curves_overlay


def _ticks(n: int = 100) -> list[MarketTick]:
    book = OrderBook(bids=[BookLevel(99.0, 1.0)], asks=[BookLevel(101.0, 1.0)], ts=0.0)
    return [MarketTick(book=book, trades=[], ts=float(i * 1_000)) for i in range(n)]


def _run_fn_with_equity(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    m = StrategyMetrics(symbol="TEST")
    slope = params.get("slope", 1.0)
    for i in range(10):
        m.equity_snapshots.append(EquitySnapshot(
            ts=float(i * 1000), realized_pnl=slope * i, unrealized_pnl=0.0,
            position=0.0, fees_paid=0.0,
        ))
    m.realized_pnl = slope * 9
    return m


def _run_fn_no_equity(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    return StrategyMetrics(symbol="TEST")


class TestEquityCurvesOverlay:
    def _results(self, n: int = 5) -> list[SweepResult]:
        grid = expand_grid(slope=[float(i) for i in range(1, n + 1)])
        return ParameterSweep.run(grid, _run_fn_with_equity, _ticks(), n_jobs=1, verbose=False)

    def test_builds_with_top_n(self):
        fig = equity_curves_overlay(self._results(), top_n=3)
        ax = fig.axes[0]
        assert len(ax.get_lines()) >= 3  # 3 series + the zero axhline

    def test_top_n_none_plots_everything(self):
        results = self._results(n=4)
        fig = equity_curves_overlay(results, top_n=None)
        ax = fig.axes[0]
        # each valid result contributes one line, plus the axhline at 0
        assert len(ax.get_lines()) == 4 + 1

    def test_custom_labels(self):
        results = self._results(n=2)
        fig = equity_curves_overlay(results, top_n=2, labels=["run A", "run B"])
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "run A" in legend_labels
        assert "run B" in legend_labels

    def test_mismatched_labels_length_raises(self):
        results = self._results(n=3)
        with pytest.raises(ValueError, match="labels has"):
            equity_curves_overlay(results, top_n=3, labels=["only one"])

    def test_no_valid_results_raises(self):
        def _failing(params, ticks):
            raise RuntimeError("boom")
        results = ParameterSweep.run([{"x": 1}], _failing, [], n_jobs=1, verbose=False)
        with pytest.raises(ValueError, match="no valid results"):
            equity_curves_overlay(results)

    def test_results_without_equity_snapshots_raises(self):
        results = ParameterSweep.run(
            expand_grid(x=[1, 2]), _run_fn_no_equity, _ticks(), n_jobs=1, verbose=False,
        )
        with pytest.raises(ValueError, match="equity_snapshots to plot"):
            equity_curves_overlay(results, top_n=None)

    def test_picks_best_by_net_pnl(self):
        results = self._results(n=5)
        fig = equity_curves_overlay(results, top_n=1, by="net_pnl")
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("slope=5" in l for l in legend_labels)  # highest slope = highest net_pnl
