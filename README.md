# realistic-mm-backtester - market making backtester

**Institutional-grade FIFO queue simulation for market making strategies.**

Most open-source backtesters fill your passive orders the moment a trade crosses your price. That's not how exchanges work. This one actually models the queue.

---

## What makes this different

**FIFO queue simulation.** When you post a bid, you join the back of the queue at that price. Trades and cancellations ahead of you reduce your position. You only get filled when the queue in front of you is exhausted. `mmbt` simulates this per-order, per-tick.

**Iceberg detection.** If a price level's visible size drops without a matching trade, the engine infers passive cancellations and advances your queue position accordingly.

**Latency modeling.** Orders and cancels don't arrive at the exchange instantly. The engine samples from a lognormal distribution (fits real co-lo latencies well) and delays execution accordingly. Your strategy sees a stale book; your orders arrive late.

**Cancel/re-quote lifecycle.** A market maker that can't cancel orders isn't making markets. `CancelOrder` is a first-class type — strategies return mixed lists of `Order` and `CancelOrder` from `on_tick`.

**Two engines, one interface.** `BacktestEngine` is fast and optimistic, use it for parameter sweeps. `ProBacktestEngine` is realistic, use it when you need to believe the results.

---

## Install

```bash
git clone https://github.com/tfrmma/realistic-mm-backtester
cd realistic-mm-backtester
pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

---

## Quick start

```python
from mmbt.core.types import BookLevel, MarketTick, OrderBook, Side, Trade
from mmbt.engine.pro import ProBacktestEngine
from mmbt.latency.config import LatencyConfig
from mmbt.queue.cancel_models import ReduceRatioCancelModel
from mmbt.risk.base import MaxInventoryRiskManager

from examples.strategies.symmetric_mm import SymmetricMM

lat = LatencyConfig(feed_us=100.0, order_us=450.0, cancel_us=280.0, jitter=0.20)
engine = ProBacktestEngine(
    latency_config=lat,
    cancel_model=ReduceRatioCancelModel(cancel_ratio=0.15),
    risk=MaxInventoryRiskManager(max_position=5.0),
    fee_rate=0.0001,
)

engine.add_strategy("mm", SymmetricMM("BTC-USD", half_spread_bps=2.0, order_size=0.1), "BTC-USD")
results = engine.run(ticks)   # list[MarketTick] from your data loader

print(results["mm"].summary())
```

Latency config also loads from YAML:

```python
lat = LatencyConfig.from_yaml("examples/configs/pro_backtest.yaml")
```

---

## Writing a strategy

Implement two methods:

```python
from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import CancelOrder, Fill, Order, OrderBook, Side, Trade
import uuid

class MyStrategy(BaseStrategy):
    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list[Order | CancelOrder]:
        # return new orders and/or cancels for existing ones
        return [
            Order(
                order_id=str(uuid.uuid4()),
                symbol="BTC-USD",
                side=Side.BUY,
                price=book.mid - 5.0,
                size=0.1,
                is_post_only=True,
            )
        ]

    def on_fill(self, fill: Fill) -> None:
        # fill.qty_in_front tells you how much queue was in front when you got hit
        # fill.queue_displacement_us tells you latency cost of a cancel that arrived late
        pass
```

`BaseStrategy` provides no-op defaults for `on_fill`, `on_start`, and `on_end`. If you don't need them, skip the base class, the engine only requires the `Strategy` Protocol.

---





## Rust hot path (S5)

The FIFO queue inner loop is ported to Rust via PyO3. The Python API is
identical — the swap is transparent.

```python
from mmbt.queue.fifo import RUST_AVAILABLE
print(RUST_AVAILABLE)   # True if the extension is compiled
```

**Build the extension:**

```bash
pip install maturin
maturin develop --release    # compiles mmbt._core in-place
```

**Performance (typical, 20 active orders):**

| Implementation | Throughput     | Speedup |
|---|---|---|
| Python pure    | ~80k ticks/s   | 1×      |
| Rust (PyO3)    | ~1.2M ticks/s  | ~15×    |

Speedup scales with active order count, more orders = more work per tick =
larger Rust advantage. Run `python benchmarks/bench_fifo.py` for your hardware.

If Rust is not installed, mmbt falls back to pure Python automatically.

**FFI boundary design:**
- `Side` as `IntEnum` (i8 in Rust)
- `BookLevel` as `(f64, f64)` tuple at the boundary
- Trade tuples `(price, size, side_int, ts)` no Python objects in the hot loop
- `f64.to_bits()` as HashMap key for price lookups (safe for finite prices)

## Multi-asset inventory (S4)

Use `Portfolio` when a strategy quotes multiple symbols:

```python
from mmbt.core.portfolio import Portfolio

class MultiSymbolMM(BaseStrategy):
    def __init__(self):
        self.portfolio = Portfolio()
        # ... per-symbol state

    def on_fill(self, fill: Fill) -> None:
        self.portfolio.apply_fill(fill, fee_rate=0.0001)

    def total_pnl(self, mids: dict) -> float:
        return self.portfolio.total_pnl(mids)
```

## Exchange adapters (S4)

Wrap CSV/Parquet data in a `CSVExchange` to match the `Exchange` protocol:

```python
from mmbt.data.exchange import CSVExchange, ExchangeMetadata

ex = CSVExchange(name="hyperliquid")
ex.register("BTC-USD", "data/btc_ticks.csv",
            ExchangeMetadata(tick_size=0.1, fee_rate_maker=-0.0001))

for tick in ex.load_ticks("BTC-USD", start_ts=0, end_ts=1e13):
    ...
```

Implement the `Exchange` protocol to add live feed or other venue adapters.

## Inventory skew strategy (S4)

`InventorySkewMM` is the second reference implementation — shifts quotes based on position:

```python
from examples.strategies.inventory_skew_mm import InventorySkewMM

strat = InventorySkewMM(
    symbol="BTC-USD",
    half_spread_bps=2.0,
    order_size=0.1,
    max_position=5.0,
    skew_bps=1.5,     # max quote shift at max_position
)
```

## Parameter sweeps (S3)

Define a module-level `run_fn` (no lambdas — they can't be pickled by `ProcessPoolExecutor`):

```python
# examples/sweep_example.py — run_fn at module level
from mmbt.core.types import MarketTick
from mmbt.reporting.metrics import StrategyMetrics
from mmbt.engine.pro import ProBacktestEngine
from examples.strategies.symmetric_mm import SymmetricMM

def run_symmetric_mm(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    engine = ProBacktestEngine(fee_rate=0.0001)
    strat  = SymmetricMM("BTC-USD", half_spread_bps=params["half_spread_bps"],
                         order_size=params["order_size"])
    engine.add_strategy("mm", strat, "BTC-USD")
    return engine.run(ticks)["mm"]
```

Then run the sweep:

```python
from mmbt.engine.sweep import ParameterSweep, expand_grid
from mmbt.reporting.sweep_plots import param_heatmap, pnl_vs_adverse, ranking_table

grid    = expand_grid(half_spread_bps=[1.0, 2.0, 3.0, 5.0], order_size=[0.05, 0.1, 0.25])
results = ParameterSweep.run(grid, run_fn=run_symmetric_mm, ticks=ticks, n_jobs=4)

df   = ParameterSweep.to_dataframe(results)        # pandas DataFrame
best = ParameterSweep.best(results, by="net_pnl")  # top result

param_heatmap(results, x_param="half_spread_bps", y_param="order_size", metric="net_pnl")
pnl_vs_adverse(results)
ranking_table(results, by="sharpe", top_n=10)
```

Use `n_jobs=1` during development to avoid subprocess overhead.

## Reporting (S2)

After a run, build a `BacktestReport` for full analysis:

```python
from mmbt.reporting.metrics import BacktestReport
from mmbt.reporting import plots

metrics  = engine.run(ticks)["mm"]
report   = BacktestReport.from_metrics(metrics, lookback_ticks=10)

report.print_summary()           # terminal stats block
plots.summary_dashboard(report)  # 2×2 matplotlib figure
plt.show()
```

Individual plots are also available standalone:

```python
plots.equity_curve(report)        # cumulative PnL, realized vs total
plots.inventory_over_time(report) # position over time
plots.adverse_selection(report)   # are we getting picked off?
plots.fill_analysis(report)       # queue depth at fill (FIFO diagnostic)
```

Load real data from CSV or Parquet:

```python
from mmbt.data import TickLoader, SyntheticConfig

loader = TickLoader.from_csv("btc_ticks.csv", symbol="BTC-USD")
# or
loader = TickLoader.from_parquet("btc_ticks.parquet", symbol="BTC-USD")
# or synthetic
loader = TickLoader.synthetic(SyntheticConfig(n_ticks=50_000, seed=42))

engine.run(loader)   # engines accept any Iterable[MarketTick]
```

CSV format (trade columns optional — leave empty for book-only ticks):
```
ts,bid_px,bid_sz,ask_px,ask_sz,trade_px,trade_sz,trade_side
1000.0,49990.0,1.5,50000.0,2.0,,,
2000.0,49985.0,1.2,49995.0,1.8,49985.0,0.5,SELL
```

## Architecture

```
mmbt/
├── core/
│   ├── types.py        — BookLevel, OrderBook, Trade, Order, CancelOrder, Fill, InventoryState
│   └── protocol.py     — Strategy, RiskManager, Exchange (typing.Protocol)
│
├── queue/
│   ├── fifo.py         — FIFOQueueState, FIFOQueueSimulator (iceberg detection, cancel support)
│   ├── cancel_models.py — ReduceRatioCancelModel, ProbQueueCancelModel
│   └── passive.py      — PassiveFillSimulator (used by BacktestEngine)
│
├── latency/
│   ├── config.py       — LatencyConfig (Pydantic, YAML/JSON serializable)
│   └── simulator.py    — LatencySimulator (min-heap event queue)
│
├── risk/
│   └── base.py         — NullRiskManager, MaxInventoryRiskManager
│
└── engine/
    ├── simple.py       — BacktestEngine (fast, heuristic fills, no latency)
    └── pro.py          — ProBacktestEngine (FIFO queue + latency, the realistic one)
```

---

## Cancel models

The cancel model controls how much of the queue ahead of you evaporates on each aggressive trade. Two implementations ship out of the box:

| Model | Behaviour | Good for |
|---|---|---|
| `ReduceRatioCancelModel(0.20)` | Fixed 20% of in-front queue cancels per trade | Fast, interpretable default |
| `ProbQueueCancelModel(min=0.05, max=0.70)` | Cancel rate scales with `trade_size / qty_in_front` | More realistic on deep books |

Implement `CancelModel` protocol to plug in your own calibration.

---

## Latency config

```yaml
# examples/configs/pro_backtest.yaml
feed_us: 100.0      # feed-to-strategy latency (median, microseconds)
order_us: 450.0     # order round-trip (median, microseconds)
cancel_us: 280.0    # cancel round-trip (median, microseconds)
jitter: 0.20        # lognormal sigma — 0.20 fits most co-lo setups
```

Latencies are sampled from a lognormal distribution per-event. Increase `jitter` to model noisier network paths.

---

## Roadmap

| Sprint | Status | Scope |
|---|---|---|
| S1 — Foundation | done | Core types, FIFO queue, cancel support, both engines |
| S2 — Reporting | done | Equity curve, inventory heatmap, adverse selection charts, TickLoader |
| S3 — Engine + Sweep | done | ParameterSweep, parallel runs, sweep plots |
| S4 — Launch | done | Exchange Protocol, Portfolio, InventorySkewMM, 80 tests, CI |
| S5 — Rust hot path | done | PyO3 port of FIFOQueueSimulator for multi-month datasets |

---

## Known limitations (S2)

- **Feed latency** is not actually applied to the book the strategy sees, it receives the current tick's book. Proper stale-book simulation requires book history; it's on the S2 list.
- **Multi-asset** inventory tracking is single-symbol for now.

---

## Contact

Built by **Taha** — algorithmic trader focused on execution and microstructure.

- Email: fadilrezokt@gmail.com
- LinkedIn: [linkedin.com/in/tahaotc](https://linkedin.com/in/tahaotc)
