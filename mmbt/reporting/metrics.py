from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from mmbt.core.types import Fill, Side


@dataclass(slots=True)
class EquitySnapshot:
    ts: float
    realized_pnl: float
    unrealized_pnl: float
    position: float
    fees_paid: float

    @property
    def equity(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid


@dataclass(slots=True)
class FillRecord:
    fill: Fill
    mid_at_fill: float


@dataclass
class StrategyMetrics:
    symbol: str = ""
    fills: list[Fill] = field(default_factory=list)
    fill_records: list[FillRecord] = field(default_factory=list)
    equity_snapshots: list[EquitySnapshot] = field(default_factory=list)
    # every tick, growing unbounded. disable for >10M ticks, you've been warned
    mid_history: list[tuple[float, float]] = field(default_factory=list)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    def net_pnl(self) -> float:
        return self.realized_pnl - self.fees_paid

    def avg_qty_in_front(self) -> float:
        maker = [f.qty_in_front for f in self.fills if f.is_maker]
        return float(np.mean(maker)) if maker else 0.0

    def summary(self) -> dict:
        return {
            "n_fills":          len(self.fills),
            "realized_pnl":     round(self.realized_pnl, 8),
            "fees_paid":        round(self.fees_paid, 8),
            "net_pnl":          round(self.net_pnl(), 8),
            "avg_qty_in_front": round(self.avg_qty_in_front(), 4),
            "equity_snapshots": len(self.equity_snapshots),
        }


@dataclass
class BacktestReport:
    metrics: StrategyMetrics
    sharpe: float
    max_drawdown: float
    win_rate: float
    adverse_scores: list[float]

    @property
    def adverse_selection_score(self) -> float:
        """Negative is good — price moved in our favor after fills."""
        return float(np.mean(self.adverse_scores)) if self.adverse_scores else 0.0

    @classmethod
    def from_metrics(
        cls,
        metrics: StrategyMetrics,
        lookback_ticks: int = 10,
    ) -> BacktestReport:
        equity_vals = [s.equity for s in metrics.equity_snapshots]
        ts_list     = [ts for ts, _ in metrics.mid_history]
        mid_list    = [m for _, m in metrics.mid_history]
        scores      = _adverse_scores(metrics.fill_records, ts_list, mid_list, lookback_ticks)
        n           = len(scores)
        return cls(
            metrics=metrics,
            sharpe=_sharpe(equity_vals),
            max_drawdown=_max_drawdown(equity_vals),
            win_rate=sum(1 for s in scores if s <= 0) / n if n else 0.0,
            adverse_scores=scores,
        )

    def print_summary(self) -> None:
        m = self.metrics
        sep = "-" * 42
        print(f"\n{sep}")
        print(f"  mmbt backtest -- {m.symbol or 'unknown'}")
        print(sep)
        rows = [
            ("fills",            len(m.fills)),
            ("realized pnl",     f"{m.realized_pnl:.8f}"),
            ("fees paid",        f"{m.fees_paid:.8f}"),
            ("net pnl",          f"{m.net_pnl():.8f}"),
            ("sharpe (raw)",     f"{self.sharpe:.4f}"),
            ("max drawdown",     f"{self.max_drawdown:.8f}"),
            ("win rate",         f"{self.win_rate:.4f}"),
            ("adverse sel.",     f"{self.adverse_selection_score:.4f}  (neg = good)"),
            ("avg qty in front", f"{m.avg_qty_in_front():.4f}"),
        ]
        for label, val in rows:
            print(f"  {label:<22} {val}")
        print(f"{sep}\n")


def _sharpe(equity_vals: list[float]) -> float:
    # not annualized, annualization needs tick frequency, caller's problem
    if len(equity_vals) < 2:
        return 0.0
    returns = np.diff(equity_vals)
    std = float(np.std(returns))
    return 0.0 if std < 1e-12 else float(np.mean(returns) / std)


def _max_drawdown(equity_vals: list[float]) -> float:
    if not equity_vals:
        return 0.0
    peak, max_dd = -np.inf, 0.0
    for e in equity_vals:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)
    return float(max_dd)


def _adverse_scores(
    fill_records: list[FillRecord],
    ts_list: list[float],
    mid_list: list[float],
    lookback_ticks: int,
) -> list[float]:
    scores: list[float] = []
    for fr in fill_records:
        idx = bisect.bisect_left(ts_list, fr.fill.ts)
        fi  = idx + lookback_ticks
        if fi >= len(mid_list):
            continue
        side_sign = 1.0 if fr.fill.side == Side.BUY else -1.0
        # positive = adversely selected (price moved against us after fill)
        scores.append((fr.mid_at_fill - mid_list[fi]) * side_sign)
    return scores
