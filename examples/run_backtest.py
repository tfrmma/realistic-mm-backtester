"""Minimal example: ProBacktestEngine + SymmetricMM on synthetic data.
Run from repo root: python examples/run_backtest.py"""

from __future__ import annotations

from mmbt.data import SyntheticConfig, TickLoader
from mmbt.engine.pro import ProBacktestEngine
from mmbt.latency.config import LatencyConfig
from mmbt.queue.cancel_models import ReduceRatioCancelModel
from mmbt.reporting.metrics import BacktestReport
from mmbt.risk.base import MaxInventoryRiskManager

from examples.strategies.symmetric_mm import SymmetricMM


def main() -> None:
    lat    = LatencyConfig(feed_us=100.0, order_us=400.0, cancel_us=250.0, jitter=0.20)
    engine = ProBacktestEngine(
        latency_config=lat,
        cancel_model=ReduceRatioCancelModel(cancel_ratio=0.15),
        risk=MaxInventoryRiskManager(max_position=5.0),
        fee_rate_maker=0.0001,
        fee_rate_taker=0.0005,
        snapshot_every=50,
        seed=42,
    )
    engine.add_strategy(
        "mm",
        SymmetricMM(symbol="BTC-USD", half_spread_bps=2.0, order_size=0.1),
        symbol="BTC-USD",
    )

    ticks   = TickLoader.synthetic(SyntheticConfig(n_ticks=10_000, seed=42)).to_list()
    metrics = engine.run(ticks)["mm"]
    report  = BacktestReport.from_metrics(metrics)
    report.print_summary()


if __name__ == "__main__":
    main()
