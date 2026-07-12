"""Grid search over half_spread_bps x order_size for SymmetricMM.
Run from repo root: python examples/sweep_example.py

run_fn must be module-level, ProcessPoolExecutor can't pickle lambdas.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from mmbt.core.types import MarketTick
from mmbt.data import SyntheticConfig, TickLoader
from mmbt.engine.pro import ProBacktestEngine
from mmbt.engine.sweep import ParameterSweep, SweepResult, expand_grid
from mmbt.latency.config import LatencyConfig
from mmbt.reporting.metrics import StrategyMetrics
from mmbt.reporting.sweep_plots import param_heatmap, pnl_vs_adverse, ranking_table

from examples.strategies.symmetric_mm import SymmetricMM


def run_symmetric_mm(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    engine = ProBacktestEngine(
        latency_config=LatencyConfig(order_us=params.get("order_us", 450.0)),
        fee_rate_maker=0.0001,
        fee_rate_taker=0.0005,
        snapshot_every=50,
    )
    strat = SymmetricMM(
        symbol="BTC-USD",
        half_spread_bps=params["half_spread_bps"],
        order_size=params["order_size"],
    )
    engine.add_strategy("mm", strat, "BTC-USD")
    return engine.run(ticks)["mm"]


def main() -> None:
    ticks = TickLoader.synthetic(SyntheticConfig(
        n_ticks=8_000, vol_per_tick=4.0, trade_prob=0.25, seed=42,
    )).to_list()

    grid = expand_grid(
        half_spread_bps=[1.0, 2.0, 3.0, 5.0],
        order_size=[0.05, 0.10, 0.25, 0.50],
    )
    print(f"{len(grid)} configs")

    results = ParameterSweep.run(grid, run_fn=run_symmetric_mm, ticks=ticks, n_jobs=4)

    df = ParameterSweep.to_dataframe(results)
    print(df.sort_values("net_pnl", ascending=False).to_string(index=False))

    top = ParameterSweep.best(results, by="net_pnl", top_n=3)
    for i, r in enumerate(top, 1):
        print(f"  {i}. {r.params}  net_pnl={r.net_pnl():.6f}  sharpe={r.sharpe():.4f}")

    param_heatmap(results, "half_spread_bps", "order_size", "net_pnl").savefig(
        "sweep_heatmap.png", dpi=150, bbox_inches="tight"
    )
    pnl_vs_adverse(results).savefig("sweep_pnl_vs_adverse.png", dpi=150, bbox_inches="tight")
    ranking_table(results, by="net_pnl", top_n=8).savefig("sweep_ranking.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("plots saved")


if __name__ == "__main__":
    main()
