"""Sweep tests run with n_jobs=1 (serial) to avoid subprocess/pickle overhead in CI."""

from __future__ import annotations

import pytest

from mmbt.core.types import BookLevel, MarketTick, OrderBook
from mmbt.engine.sweep import (
    OutOfSampleResult,
    WalkForwardFold,
    expand_grid,
    out_of_sample_validate,
    walk_forward_summary,
    walk_forward_validate,
)
from mmbt.reporting.metrics import StrategyMetrics


def _regime_ticks(n: int = 1000, switch_at: float = 5000.0) -> list[MarketTick]:
    """ts runs 0, 10, 20, ... -- avg_ts < switch_at is 'regime A', >= is 'regime B'."""
    book = OrderBook(bids=[BookLevel(99.0, 1.0)], asks=[BookLevel(101.0, 1.0)], ts=0.0)
    return [MarketTick(book=book, trades=[], ts=float(i * 10)) for i in range(n)]


def _regime_run_fn(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    """x=1 wins in regime A (early ticks), x=2 wins in regime B (late ticks)
    a deliberately overfittable signal so the tests can check the detection
    logic against a known-correct answer."""
    m = StrategyMetrics(symbol="T")
    avg_ts   = sum(t.ts for t in ticks) / len(ticks)
    regime_a = avg_ts < 5000.0
    x = params["x"]
    if regime_a:
        m.realized_pnl = 100.0 if x == 1 else -50.0
    else:
        m.realized_pnl = 100.0 if x == 2 else -50.0
    return m


def _flat_run_fn(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    m = StrategyMetrics(symbol="T")
    m.realized_pnl = float(params.get("x", 0))
    return m


def _failing_run_fn(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    raise RuntimeError("intentional")


class TestOutOfSampleValidate:
    def test_detects_known_overfit(self):
        ticks   = _regime_ticks()
        grid    = expand_grid(x=[1, 2])
        results = out_of_sample_validate(grid, _regime_run_fn, ticks, train_frac=0.8,
                                          top_n=2, n_jobs=1, verbose=False)
        by_x = {r.params["x"]: r for r in results}
        assert by_x[1].in_sample_pnl == pytest.approx(100.0)
        assert by_x[1].out_of_sample_pnl == pytest.approx(-50.0)
        assert by_x[1].overfit_flag is True
        assert by_x[2].out_of_sample_pnl == pytest.approx(100.0)
        assert by_x[2].overfit_flag is False

    def test_degradation_sign(self):
        ticks   = _regime_ticks()
        grid    = expand_grid(x=[1, 2])
        results = out_of_sample_validate(grid, _regime_run_fn, ticks, train_frac=0.8,
                                          top_n=2, n_jobs=1, verbose=False)
        by_x = {r.params["x"]: r for r in results}
        assert by_x[1].degradation == pytest.approx(150.0)   # 100 - (-50), got much worse OOS
        assert by_x[2].degradation == pytest.approx(-150.0)  # -50 - 100, got much better OOS

    def test_train_frac_out_of_bounds_raises(self):
        ticks = _regime_ticks(n=100)
        grid  = expand_grid(x=[1])
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="train_frac"):
                out_of_sample_validate(grid, _flat_run_fn, ticks, train_frac=bad, n_jobs=1, verbose=False)

    def test_too_few_ticks_for_split_raises(self):
        ticks = _regime_ticks(n=1)  # any split leaves one side empty
        grid  = expand_grid(x=[1])
        with pytest.raises(ValueError):
            out_of_sample_validate(grid, _flat_run_fn, ticks, train_frac=0.5, n_jobs=1, verbose=False)

    def test_empty_configs_returns_empty(self):
        ticks = _regime_ticks(n=100)
        assert out_of_sample_validate([], _flat_run_fn, ticks, n_jobs=1, verbose=False) == []

    def test_all_configs_fail_returns_empty(self):
        ticks = _regime_ticks(n=100)
        grid  = expand_grid(x=[1, 2])
        assert out_of_sample_validate(grid, _failing_run_fn, ticks, n_jobs=1, verbose=False) == []

    def test_top_n_respected(self):
        ticks   = _regime_ticks(n=200)
        grid    = expand_grid(x=[1, 2, 3, 4])
        results = out_of_sample_validate(grid, _flat_run_fn, ticks, top_n=2, n_jobs=1, verbose=False)
        assert len(results) == 2

    def test_result_type(self):
        ticks   = _regime_ticks(n=200)
        results = out_of_sample_validate(expand_grid(x=[1]), _flat_run_fn, ticks, n_jobs=1, verbose=False)
        assert isinstance(results[0], OutOfSampleResult)


class TestWalkForwardValidate:
    def test_n_folds_must_be_positive(self):
        ticks = _regime_ticks(n=100)
        with pytest.raises(ValueError, match="n_folds"):
            walk_forward_validate(expand_grid(x=[1]), _flat_run_fn, ticks, n_folds=0, n_jobs=1, verbose=False)

    def test_not_enough_ticks_raises(self):
        ticks = _regime_ticks(n=3)
        with pytest.raises(ValueError):
            walk_forward_validate(expand_grid(x=[1]), _flat_run_fn, ticks, n_folds=10, n_jobs=1, verbose=False)

    def test_returns_n_folds(self):
        ticks = _regime_ticks(n=1000)
        folds = walk_forward_validate(expand_grid(x=[1, 2]), _regime_run_fn, ticks,
                                       n_folds=5, n_jobs=1, verbose=False)
        assert len(folds) == 5
        assert all(isinstance(f, WalkForwardFold) for f in folds)

    def test_last_fold_absorbs_remainder(self):
        # 1000 ticks / (5+1) = 166 per chunk, remainder of 10 ticks
        ticks = _regime_ticks(n=1000)
        folds = walk_forward_validate(expand_grid(x=[1]), _flat_run_fn, ticks,
                                       n_folds=5, n_jobs=1, verbose=False)
        assert folds[-1].test_range[1] == 1000  # last fold's test end reaches the very end

    def test_rolling_window_is_fixed_size(self):
        ticks = _regime_ticks(n=1000)
        folds = walk_forward_validate(expand_grid(x=[1]), _flat_run_fn, ticks,
                                       n_folds=5, expanding=False, n_jobs=1, verbose=False)
        sizes = {f.train_range[1] - f.train_range[0] for f in folds}
        assert len(sizes) == 1  # every fold's training window is the same size

    def test_expanding_window_grows(self):
        ticks = _regime_ticks(n=1000)
        folds = walk_forward_validate(expand_grid(x=[1]), _flat_run_fn, ticks,
                                       n_folds=5, expanding=True, n_jobs=1, verbose=False)
        sizes = [f.train_range[1] - f.train_range[0] for f in folds]
        assert sizes == sorted(sizes)          # strictly non-decreasing
        assert sizes[0] < sizes[-1]            # and genuinely grows
        assert all(f.train_range[0] == 0 for f in folds)  # always starts from the beginning

    def test_detects_regime_shift_fold(self):
        # the switch from regime A to B happens at ts=5000 -> tick index 500
        # (ts = i*10). With n_folds=5 over 1000 ticks, chunk=166, so the test
        # window containing index 500 is fold 3 (test[498:664]) that's the
        # fold where the config trained on (still regime-A) history should
        # underperform against the OOS window that straddles the regime switch.
        ticks = _regime_ticks(n=1000)
        folds = walk_forward_validate(expand_grid(x=[1, 2]), _regime_run_fn, ticks,
                                       n_folds=5, n_jobs=1, verbose=False)
        assert folds[2].out_of_sample_pnl == pytest.approx(-50.0)
        # folds safely on one side of the switch should be profitable OOS
        assert folds[0].out_of_sample_pnl == pytest.approx(100.0)
        assert folds[4].out_of_sample_pnl == pytest.approx(100.0)

    def test_summary_stats_match_known_scenario(self):
        ticks   = _regime_ticks(n=1000)
        folds   = walk_forward_validate(expand_grid(x=[1, 2]), _regime_run_fn, ticks,
                                         n_folds=5, n_jobs=1, verbose=False)
        summary = walk_forward_summary(folds)
        assert summary["n_folds"] == 5
        assert summary["oos_profitable_folds"] == 4  # all but the regime-switch fold
        assert summary["oos_profitable_frac"] == pytest.approx(0.8)

    def test_summary_empty_folds(self):
        assert walk_forward_summary([]) == {"n_folds": 0}

    def test_empty_configs_returns_empty_folds(self):
        ticks = _regime_ticks(n=1000)
        folds = walk_forward_validate([], _flat_run_fn, ticks, n_folds=3, n_jobs=1, verbose=False)
        assert folds == []
