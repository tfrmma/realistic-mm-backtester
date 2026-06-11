"""FIFO queue benchmark: Python pure vs Rust extension.

Run from repo root:
    python benchmarks/bench_fifo.py

Build Rust extension first for the interesting comparison:
    maturin develop --release
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from mmbt.core.types import BookLevel, MarketTick, Order, OrderBook, Side
from mmbt.data import SyntheticConfig, TickLoader
from mmbt.queue.cancel_models import ReduceRatioCancelModel
from mmbt.queue.fifo import RUST_AVAILABLE, FIFOQueueSimulator

N_TICKS        = 500_000
N_ACTIVE       = 20
CANCEL_RATIO   = 0.15


@dataclass
class BenchResult:
    label:     str
    n_ticks:   int
    n_fills:   int
    elapsed_s: float

    @property
    def ticks_per_s(self) -> float:
        return self.n_ticks / self.elapsed_s

    def __str__(self) -> str:
        return (
            f"  {self.label:<18}  {self.elapsed_s:>6.2f}s  "
            f"{self.ticks_per_s:>12,.0f} ticks/s  "
            f"{self.n_fills:>6} fills"
        )


def _order(side: Side, price: float) -> Order:
    return Order(order_id=str(uuid.uuid4()), symbol="BTC-USD",
                 side=side, price=price, size=0.1, ts=0.0)


def _run(ticks: list[MarketTick], use_rust: bool) -> BenchResult:
    label  = "Rust" if (use_rust and RUST_AVAILABLE) else "Python (pure)"
    model  = ReduceRatioCancelModel(CANCEL_RATIO)
    sim    = FIFOQueueSimulator(model, use_rust=use_rust)

    first  = ticks[0].book
    for _ in range(N_ACTIVE // 2):
        sim.register(_order(Side.BUY,  first.best_bid.price), first)
        sim.register(_order(Side.SELL, first.best_ask.price), first)

    n_fills = 0
    t0 = time.perf_counter()

    for tick in ticks:
        book    = tick.book
        fills   = sim.process_tick(book, tick.trades)
        n_fills += len(fills)
        # keep N_ACTIVE orders in the queue so the bench measures real work
        needed = N_ACTIVE - sim.pending_count()
        for _ in range(max(0, needed // 2)):
            sim.register(_order(Side.BUY,  book.best_bid.price), book)
            sim.register(_order(Side.SELL, book.best_ask.price), book)

    elapsed = time.perf_counter() - t0
    return BenchResult(label=label, n_ticks=len(ticks), n_fills=n_fills, elapsed_s=elapsed)


def validate(ticks: list[MarketTick]) -> None:
    if not RUST_AVAILABLE:
        print("  [skip] Rust not compiled")
        return
    r_py = _run(ticks[:5_000], use_rust=False)
    r_rs = _run(ticks[:5_000], use_rust=True)
    if r_py.n_fills == r_rs.n_fills:
        print(f"  fill counts match: {r_py.n_fills}")
    else:
        print(f"  MISMATCH — Python: {r_py.n_fills}, Rust: {r_rs.n_fills}")


def main() -> None:
    print(f"\nmmbt FIFO benchmark  ({N_TICKS:,} ticks, {N_ACTIVE} active orders)")
    print(f"Rust ext: {'yes' if RUST_AVAILABLE else 'no  (maturin develop --release to build)'}\n")

    ticks = TickLoader.synthetic(
        SyntheticConfig(n_ticks=N_TICKS, vol_per_tick=5.0, trade_prob=0.35, seed=0)
    ).to_list()

    print("validating correctness...")
    validate(ticks)
    print()

    sep = "-" * 62
    print(sep)
    print(f"  {'impl':<18}  {'time':>6}   {'ticks/s':>12}   {'fills':>6}")
    print(sep)

    print(_run(ticks, use_rust=False))

    if RUST_AVAILABLE:
        r_rs    = _run(ticks, use_rust=True)
        r_py    = _run(ticks, use_rust=False)
        print(r_rs)
        print(sep)
        print(f"\n  speedup: {r_py.elapsed_s / r_rs.elapsed_s:.1f}x\n")
    else:
        print(sep)
        print("\n  maturin develop --release  to see the Rust path\n")


if __name__ == "__main__":
    main()
