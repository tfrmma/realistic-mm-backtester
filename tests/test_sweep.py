"""Sweep tests run with n_jobs=1 (serial) to avoid subprocess/pickle overhead in CI."""

from __future__ import annotations

import pytest

from mmbt.core.types import BookLevel, MarketTick, OrderBook
from mmbt.engine.sweep import ParameterSweep, SweepResult, expand_grid
from mmbt.reporting.metrics import StrategyMetrics


def _ticks(n: int = 100) -> list[MarketTick]:
    book = OrderBook(bids=[BookLevel(99.0, 1.0)], asks=[BookLevel(101.0, 1.0)], ts=0.0)
    return [MarketTick(book=book, trades=[], ts=float(i * 1_000)) for i in range(n)]


def _dummy_run(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    m = StrategyMetrics(symbol="TEST")
    m.realized_pnl = params.get("half_spread_bps", 0.0)
    return m


def _failing_run(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    raise RuntimeError("intentional")


def _sometimes_fail(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    if params.get("fail", False):
        raise RuntimeError("as instructed")
    m = StrategyMetrics(symbol="TEST")
    m.realized_pnl = 1.0
    return m


class TestExpandGrid:
    def test_cartesian(self):
        grid = expand_grid(x=[1, 2], y=["a", "b"])
        assert len(grid) == 4
        assert {"x": 1, "y": "a"} in grid

    def test_single_param(self):
        assert len(expand_grid(x=[10, 20, 30])) == 3

    def test_triple(self):
        assert len(expand_grid(a=[1, 2], b=[3, 4], c=[5, 6])) == 8

    def test_empty_list(self):
        assert expand_grid(x=[], y=[1, 2]) == []


class TestSweepRun:
    def test_runs_all(self):
        r = ParameterSweep.run(expand_grid(x=[1.0, 2.0, 3.0]), _dummy_run, _ticks(), n_jobs=1, verbose=False)
        assert len(r) == 3

    def test_empty_configs(self):
        assert ParameterSweep.run([], _dummy_run, [], n_jobs=1, verbose=False) == []

    def test_error_captured(self):
        r = ParameterSweep.run([{"x": 1}], _failing_run, [], n_jobs=1, verbose=False)
        assert r[0].is_valid is False
        assert r[0].error is not None

    def test_partial_failure(self):
        configs = [{"fail": False, "id": 1}, {"fail": True, "id": 2}, {"fail": False, "id": 3}]
        r = ParameterSweep.run(configs, _sometimes_fail, _ticks(), n_jobs=1, verbose=False)
        assert sum(1 for x in r if x.is_valid) == 2
        assert sum(1 for x in r if not x.is_valid) == 1

    def test_elapsed_populated(self):
        r = ParameterSweep.run([{"x": 1.0}], _dummy_run, _ticks(), n_jobs=1, verbose=False)
        assert r[0].elapsed_s >= 0.0


class TestSweepBest:
    def _results(self):
        return ParameterSweep.run(
            expand_grid(half_spread_bps=[1.0, 2.0, 3.0]),
            _dummy_run, _ticks(), n_jobs=1, verbose=False,
        )

    def test_best_net_pnl(self):
        best = ParameterSweep.best(self._results(), by="net_pnl", top_n=1)
        assert best[0].params["half_spread_bps"] == pytest.approx(3.0)

    def test_top_n(self):
        assert len(ParameterSweep.best(self._results(), top_n=2)) == 2

    def test_top_n_larger_than_results(self):
        r = self._results()
        assert len(ParameterSweep.best(r, top_n=100)) == len([x for x in r if x.is_valid])

    def test_all_failed(self):
        r = ParameterSweep.run([{"x": 1}], _failing_run, [], n_jobs=1, verbose=False)
        assert ParameterSweep.best(r) == []

    def test_bad_key_raises(self):
        with pytest.raises(ValueError, match="unknown sort key"):
            ParameterSweep.best(self._results(), by="not_real")


class TestToDataframe:
    def test_shape(self):
        pytest.importorskip("pandas")
        r  = ParameterSweep.run(expand_grid(x=[1.0, 2.0]), _dummy_run, _ticks(), n_jobs=1, verbose=False)
        df = ParameterSweep.to_dataframe(r)
        assert len(df) == 2
        assert "x" in df.columns and "net_pnl" in df.columns

    def test_error_rows(self):
        pytest.importorskip("pandas")
        r  = ParameterSweep.run([{"x": 1}], _failing_run, [], n_jobs=1, verbose=False)
        df = ParameterSweep.to_dataframe(r)
        assert "error" in df.columns
