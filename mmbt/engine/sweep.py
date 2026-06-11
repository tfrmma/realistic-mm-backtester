from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

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


# module-level worker — must be here to survive pickle
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
