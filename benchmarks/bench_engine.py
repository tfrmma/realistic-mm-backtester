"""Full ProBacktestEngine throughput benchmark.
Run from repo root: python benchmarks/bench_engine.py"""

from __future__ import annotations

import time

from mmbt.data import SyntheticConfig, TickLoader
from mmbt.engine.pro import ProBacktestEngine
from mmbt.latency.config import LatencyConfig
from mmbt.queue.cancel_models import ReduceRatioCancelModel
from mmbt.queue.fifo import RUST_AVAILABLE

from examples.strategies.inventory_skew_mm import InventorySkewMM


def _engine() -> ProBacktestEngine:
    return ProBacktestEngine(
        latency_config=LatencyConfig(order_us=400.0, cancel_us=250.0, jitter=0.15),
        cancel_model=ReduceRatioCancelModel(0.15),
        fee_rate=0.0001, snapshot_every=500, seed=42,
    )


def bench(n_ticks: int) -> tuple[float, float]:
    ticks = TickLoader.synthetic(
        SyntheticConfig(n_ticks=n_ticks, vol_per_tick=4.0, trade_prob=0.30, seed=0)
    ).to_list()
    engine = _engine()
    engine.add_strategy(
        "mm",
        InventorySkewMM("BTC-USD", half_spread_bps=2.0, order_size=0.1,
                        max_position=5.0, skew_bps=1.5),
        "BTC-USD",
    )
    t0 = time.perf_counter()
    engine.run(ticks)
    elapsed = time.perf_counter() - t0
    return elapsed, n_ticks / elapsed


def main() -> None:
    print(f"\nmmbt ProBacktestEngine throughput  (Rust ext: {'yes' if RUST_AVAILABLE else 'no'})\n")
    sep = "-" * 52
    print(sep)
    print(f"  {'ticks':>10}   {'time':>7}   {'ticks/s':>14}")
    print(sep)
    for n in [10_000, 100_000, 500_000, 1_000_000]:
        try:
            elapsed, rate = bench(n)
            print(f"  {n:>10,}   {elapsed:>6.2f}s   {rate:>12,.0f} t/s")
        except MemoryError:
            print(f"  {n:>10,}   OOM")
            break
    print(sep)
    print()


if __name__ == "__main__":
    main()
