from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

import numpy as np

from mmbt.core.types import MarketTick
from mmbt.reporting.metrics import BacktestReport, StrategyMetrics

RunFn = Callable[[dict[str, Any], list[MarketTick]], StrategyMetrics]


@dataclass
class SweepResult:
    params: dict[str, Any]
    metrics: StrategyMetrics
    report: BacktestReport | None = None
    elapsed_s: float = 0.0
    error: Exception | None = None

    @property
    def is_valid(self) -> bool:
        return self.error is None

    def net_pnl(self) -> float:
        return self.metrics.net_pnl() if self.is_valid else float("-inf")

    def sharpe(self) -> float:
        return self.report.sharpe if (self.is_valid and self.report) else 0.0


@dataclass
class OutOfSampleResult:
    """A config's in-sample (training) result paired with its out-of-sample
    (held-out) result the comparison is the whole point of running this."""
    params: dict[str, Any]
    in_sample: SweepResult
    out_of_sample: SweepResult

    @property
    def in_sample_pnl(self) -> float:
        return self.in_sample.net_pnl()

    @property
    def out_of_sample_pnl(self) -> float:
        return self.out_of_sample.net_pnl()

    @property
    def degradation(self) -> float:
        """in-sample minus out-of-sample net_pnl. Positive = OOS worse, the
        usual overfitting direction. Negative means it did *better* held out,
        which happens too (small samples, favorable OOS regime) not
        itself a red flag."""
        return self.in_sample_pnl - self.out_of_sample_pnl

    @property
    def overfit_flag(self) -> bool:
        """Cheap red flag, not a rigorous test: looked profitable in
        training, lost money out of sample. A config can fail this and still
        be fine (regime shift, thin OOS sample) use degradation and your
        own judgement too, don't just filter on this bool."""
        return self.in_sample_pnl > 0.0 and self.out_of_sample_pnl <= 0.0


@dataclass
class WalkForwardFold:
    """One train-then-test step of a walk-forward run. train_range/test_range
    are tick-index slices (start, end) into the original ticks list."""
    fold_index: int
    train_range: tuple[int, int]
    test_range: tuple[int, int]
    best_params: dict[str, Any]
    in_sample: SweepResult
    out_of_sample: SweepResult

    @property
    def out_of_sample_pnl(self) -> float:
        return self.out_of_sample.net_pnl()


# module-level worker must be here to survive pickle
def _worker(args: tuple[dict, RunFn, list[MarketTick]]) -> SweepResult:
    params, run_fn, ticks = args
    t0 = time.perf_counter()
    try:
        metrics = run_fn(params, ticks)
        return SweepResult(
            params=params,
            metrics=metrics,
            report=BacktestReport.from_metrics(metrics),
            elapsed_s=time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return SweepResult(
            params=params,
            metrics=StrategyMetrics(),
            error=exc,
            elapsed_s=time.perf_counter() - t0,
        )


def expand_grid(**kwargs: list[Any]) -> list[dict[str, Any]]:
    """Cartesian product. expand_grid(x=[1,2], y=[3,4]) -> 4 configs."""
    keys = list(kwargs.keys())
    return [dict(zip(keys, combo)) for combo in product(*kwargs.values())]


class ParameterSweep:
    """
    Grid search in parallel via ProcessPoolExecutor.
    run_fn must be module-level (picklable) no lambdas.
    Use n_jobs=1 for quick tests to avoid subprocess overhead.
    """

    @staticmethod
    def run(
        configs: list[dict[str, Any]],
        run_fn: RunFn,
        ticks: list[MarketTick],
        n_jobs: int = -1,
        verbose: bool = True,
    ) -> list[SweepResult]:
        if not configs:
            return []

        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        args = [(p, run_fn, ticks) for p in configs]

        if n_workers == 1 or len(configs) == 1:
            results = []
            for i, a in enumerate(args, 1):
                r = _worker(a)
                results.append(r)
                if verbose:
                    _log(i, len(configs), r)
            return results

        results = []
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, a): a[0] for a in args}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                done += 1
                if verbose:
                    _log(done, len(configs), r)

        if verbose:
            _print_summary(results)
        return results

    @staticmethod
    def best(
        results: list[SweepResult],
        by: str = "net_pnl",
        top_n: int = 1,
    ) -> list[SweepResult]:
        valid = [r for r in results if r.is_valid]
        if not valid:
            return []
        _keys: dict[str, Callable[[SweepResult], float]] = {
            "net_pnl":     lambda r: r.net_pnl(),
            "sharpe":      lambda r: r.sharpe(),
            "win_rate":    lambda r: r.report.win_rate if r.report else 0.0,
            "adverse_sel": lambda r: -(r.report.adverse_selection_score if r.report else 0.0),
        }
        if by not in _keys:
            raise ValueError(f"unknown sort key '{by}'. choose from: {list(_keys)}")
        return sorted(valid, key=_keys[by], reverse=True)[:top_n]

    @staticmethod
    def to_dataframe(results: list[SweepResult]) -> object:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install realistic-mm-backtester[dev]")

        rows = []
        for r in results:
            row: dict[str, Any] = dict(r.params)
            if r.is_valid:
                row.update(r.metrics.summary())
                if r.report:
                    row["sharpe"]       = round(r.report.sharpe, 6)
                    row["max_drawdown"] = round(r.report.max_drawdown, 8)
                    row["win_rate"]     = round(r.report.win_rate, 6)
                    row["adverse_sel"]  = round(r.report.adverse_selection_score, 6)
            else:
                row["error"] = str(r.error)
            row["elapsed_s"] = round(r.elapsed_s, 3)
            rows.append(row)
        return pd.DataFrame(rows)


def _log(done: int, total: int, r: SweepResult) -> None:
    w = len(str(total))
    status = (
        f"net_pnl={r.net_pnl():.6f}  ({r.elapsed_s:.2f}s)"
        if r.is_valid
        else f"ERROR: {r.error}"
    )
    print(f"  [{done:>{w}}/{total}] {r.params}  ->  {status}")


def _print_summary(results: list[SweepResult]) -> None:
    n_ok  = sum(1 for r in results if r.is_valid)
    n_err = len(results) - n_ok
    print(f"\nsweep done: {n_ok}/{len(results)} ok" + (f", {n_err} failed" if n_err else ""))
    best = ParameterSweep.best(results, by="net_pnl")
    if best:
        b = best[0]
        print(f"best net_pnl={b.net_pnl():.6f}  @  {b.params}\n")


def _train_test_sweep(
    configs: list[dict[str, Any]],
    run_fn: RunFn,
    train_ticks: list[MarketTick],
    test_ticks: list[MarketTick],
    top_n: int,
    by: str,
    n_jobs: int,
    verbose: bool,
) -> list[OutOfSampleResult]:
    """Shared core of out_of_sample_validate and walk_forward_validate: sweep
    the full grid on train_ticks, take the top_n by `by`, then re-run just
    those configs on test_ticks so each has a genuine held-out result."""
    in_sample_results = ParameterSweep.run(configs, run_fn, train_ticks, n_jobs=n_jobs, verbose=verbose)
    best = ParameterSweep.best(in_sample_results, by=by, top_n=top_n)
    if not best:
        return []
    oos_results = ParameterSweep.run([b.params for b in best], run_fn, test_ticks, n_jobs=n_jobs, verbose=verbose)
    return [
        OutOfSampleResult(params=b.params, in_sample=b, out_of_sample=oos)
        for b, oos in zip(best, oos_results)
    ]


def out_of_sample_validate(
    configs: list[dict[str, Any]],
    run_fn: RunFn,
    ticks: list[MarketTick],
    train_frac: float = 0.8,
    top_n: int = 5,
    by: str = "net_pnl",
    n_jobs: int = -1,
    verbose: bool = True,
) -> list[OutOfSampleResult]:
    """
    Chronological train/test split no shuffling, this is time-series data
    and shuffling would leak future information into training. Runs the full
    grid on the training slice, picks the top_n by `by`, then re-runs ONLY
    those configs on the held-out test slice. A config that looks great
    in-sample but falls apart out-of-sample is overfit to that specific
    stretch of history, not a real edge.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    split_idx = int(len(ticks) * train_frac)
    if split_idx <= 0 or split_idx >= len(ticks):
        raise ValueError(f"train_frac={train_frac} leaves an empty train or test slice for {len(ticks)} ticks")
    train_ticks, test_ticks = ticks[:split_idx], ticks[split_idx:]

    if verbose:
        print(f"out-of-sample validation: {len(train_ticks)} train ticks / {len(test_ticks)} test ticks")
        print("--- in-sample (training) sweep ---")
    results = _train_test_sweep(configs, run_fn, train_ticks, test_ticks, top_n, by, n_jobs, verbose)
    if verbose:
        if results:
            print(f"\n--- out-of-sample validation of top {len(results)} configs ---")
        _print_oos_summary(results)
    return results


def walk_forward_validate(
    configs: list[dict[str, Any]],
    run_fn: RunFn,
    ticks: list[MarketTick],
    n_folds: int = 5,
    expanding: bool = False,
    by: str = "net_pnl",
    n_jobs: int = -1,
    verbose: bool = True,
) -> list[WalkForwardFold]:
    """
    Rolls a train/test window forward through the tick history in n_folds
    steps, re-optimizing on each fold's training slice and validating on the
    immediately following out-of-sample slice. A single static train/test
    split (out_of_sample_validate) can get lucky or unlucky landing in one
    particular regime; walking forward through several folds is a sturdier
    check that a config's edge isn't an artifact of one slice of history
    more relevant for microstructure strategies than a static split, since
    regimes (vol, spread, flow toxicity) shift within a single day.

    expanding=False (default): fixed-size rolling window the training
    window slides forward, same length each fold, only recent history.
    expanding=True: the training window grows from the start each fold,
    using all history seen so far.

    Splits `ticks` into n_folds+1 equal-sized contiguous chunks: fold i
    trains on chunk i (or chunks 0..i if expanding) and tests on chunk i+1.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    n = len(ticks)
    chunk = n // (n_folds + 1)
    if chunk == 0:
        raise ValueError(f"not enough ticks ({n}) to form {n_folds} folds")

    folds: list[WalkForwardFold] = []
    for i in range(n_folds):
        test_start  = (i + 1) * chunk
        test_end    = n if i == n_folds - 1 else test_start + chunk  # last fold absorbs any remainder
        train_start = 0 if expanding else i * chunk
        train_end   = test_start
        train_ticks = ticks[train_start:train_end]
        test_ticks  = ticks[test_start:test_end]

        if verbose:
            print(f"\n=== fold {i + 1}/{n_folds} === "
                  f"train[{train_start}:{train_end}] ({len(train_ticks)} ticks) -> "
                  f"test[{test_start}:{test_end}] ({len(test_ticks)} ticks)")
        results = _train_test_sweep(configs, run_fn, train_ticks, test_ticks, top_n=1, by=by,
                                     n_jobs=n_jobs, verbose=verbose)
        if results:
            r = results[0]
            folds.append(WalkForwardFold(
                fold_index=i, train_range=(train_start, train_end), test_range=(test_start, test_end),
                best_params=r.params, in_sample=r.in_sample, out_of_sample=r.out_of_sample,
            ))

    if verbose:
        _print_walk_forward_summary(folds)
    return folds


def walk_forward_summary(folds: list[WalkForwardFold]) -> dict[str, Any]:
    """Aggregate stats across folds: how often OOS was profitable, how
    volatile OOS pnl was fold-to-fold, and how consistently the SAME config
    kept winning (a config that wins a different fold every time is a
    coin flip, not a strategy)."""
    if not folds:
        return {"n_folds": 0}
    oos_pnls = [f.out_of_sample_pnl for f in folds]
    profitable = sum(1 for p in oos_pnls if p > 0)

    param_counts: dict[str, int] = {}
    for f in folds:
        key = str(sorted(f.best_params.items()))
        param_counts[key] = param_counts.get(key, 0) + 1
    _, most_common_count = max(param_counts.items(), key=lambda kv: kv[1])

    return {
        "n_folds": len(folds),
        "oos_profitable_folds": profitable,
        "oos_profitable_frac": profitable / len(folds),
        "avg_oos_net_pnl": float(np.mean(oos_pnls)),
        "std_oos_net_pnl": float(np.std(oos_pnls)),
        "param_consistency": most_common_count / len(folds),
    }


def _print_oos_summary(results: list[OutOfSampleResult]) -> None:
    if not results:
        print("no valid in-sample results to validate")
        return
    print(f"\n{'params':<45} {'in-sample':>12} {'out-of-sample':>14} {'degradation':>12}")
    for r in results:
        flag = "  [OVERFIT?]" if r.overfit_flag else ""
        print(f"{str(r.params):<45} {r.in_sample_pnl:>12.4f} {r.out_of_sample_pnl:>14.4f} "
              f"{r.degradation:>12.4f}{flag}")
    print()


def _print_walk_forward_summary(folds: list[WalkForwardFold]) -> None:
    if not folds:
        print("no valid folds")
        return
    print(f"\n=== walk-forward summary ({len(folds)} folds) ===")
    for f in folds:
        print(f"  fold {f.fold_index + 1}: best={f.best_params}  oos_net_pnl={f.out_of_sample_pnl:.4f}")
    s = walk_forward_summary(folds)
    print(f"\n  OOS profitable folds: {s['oos_profitable_folds']}/{s['n_folds']} ({s['oos_profitable_frac']:.0%})")
    print(f"  avg OOS net_pnl: {s['avg_oos_net_pnl']:.4f}  (std {s['std_oos_net_pnl']:.4f})")
    print(f"  winning-config consistency: {s['param_consistency']:.0%}\n")
