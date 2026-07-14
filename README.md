# realistic-mm-backtester - market making backtester

**FIFO queue simulation for market making strategies.**

Most open-source backtesters fill your passive orders the moment a trade crosses your price. That's not how exchanges work. This one actually models the queue.

---

## What makes this different

**FIFO queue simulation.** When you post a bid, you join the back of the queue at that price. Trades and cancellations ahead of you reduce your position. You only get filled when the queue in front of you is exhausted. `mmbt` simulates this per-order, per-tick.

**Iceberg detection.** If a price level's visible size drops without a matching trade, the engine infers passive cancellations and advances your queue position accordingly.

**Latency modeling.** Orders and cancels don't arrive at the exchange instantly. The engine samples from a lognormal distribution (fits real co-lo latencies well) and delays execution accordingly. Your strategy sees a stale book; your orders arrive late.

**Cancel/re-quote lifecycle.** A market maker that can't cancel orders isn't making markets. `CancelOrder` is a first-class type - strategies return mixed lists of `Order` and `CancelOrder` from `on_tick`.

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
    fee_rate_maker=0.0001,
    fee_rate_taker=0.0005,
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

## Taker orders

An `Order` whose price already crosses the book (a BUY at or above the best
ask, a SELL at or below the best bid) doesn't join the passive queue - it's
handled as a taker order instead:

- `is_post_only=False` (default): fills immediately against visible depth,
  walking multiple book levels if needed for a size-weighted average price
  (see `queue/taker.py`). Any unfilled remainder does **not** rest - same IOC
  semantics as a real market/marketable-limit order.
- `is_post_only=True`: rejected outright, same as a real exchange would
  reject a post-only order that would take liquidity. Check
  `StrategyMetrics.rejected_orders` to see how often this happened.

In `ProBacktestEngine` this is evaluated against the **true** book at the
moment the order actually lands (after `order_us` latency) - not the stale
view the strategy saw when it decided to submit. A resting-looking order can
still get taker-filled or rejected if the market moved during the trip.

```python
Order(order_id=str(uuid.uuid4()), symbol="BTC-USD", side=Side.BUY,
      price=book.best_ask.price, size=0.05, is_post_only=False)  # sweeps the ask
```

---





## Rust hot path (S5)

The FIFO queue inner loop is ported to Rust via PyO3. The Python API is
identical - the swap is transparent. Both `ReduceRatioCancelModel` and
`ProbQueueCancelModel` are accelerated; a custom `CancelModel` subclass falls
back to pure Python automatically (Rust can't call back into arbitrary
Python cancellation logic) - check `FIFOQueueSimulator.using_rust` if you
need to confirm which path a given run actually took.

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
larger Rust advantage. Run `python benchmarks/bench_fifo.py` for your hardware
and cancel model of choice.

If Rust is not installed, mmbt falls back to pure Python automatically.

**FFI boundary design:**
- `Side` as `IntEnum` (i8 in Rust)
- `BookLevel` as `(f64, f64)` tuple at the boundary
- Trade tuples `(price, size, side_int, ts)` no Python objects in the hot loop
- `f64.to_bits()` as HashMap key for price lookups (safe for finite prices)
- Cancel model is a Rust-side enum (`ReduceRatio { ratio }` /
  `ProbQueue { min_ratio, max_ratio }`) chosen at construction time, mirroring
  the two Python `CancelModel` implementations - see `queue/fifo.py`'s
  `_build_rust_core()` for the Python -> Rust mapping.

## Multi-asset inventory (S4)

Use `Portfolio` when a strategy quotes multiple symbols but still runs on a
single-symbol engine (one `BacktestEngine`/`ProBacktestEngine` per symbol) and
just wants combined PnL/position tracking:

```python
from mmbt.core.portfolio import Portfolio

class MultiSymbolMM(BaseStrategy):
    def __init__(self):
        self.portfolio = Portfolio()
        # ... per-symbol state

    def on_fill(self, fill: Fill) -> None:
        self.portfolio.apply_fill(fill, fee_rate_maker=0.0001, fee_rate_taker=0.0005)

    def total_pnl(self, mids: dict) -> float:
        return self.portfolio.total_pnl(mids)
```

If you need a *single* strategy instance to actually react across symbols in
one decision (hedging, correlated quoting, portfolio-level risk) rather than
just aggregating PnL after the fact, use `MultiAssetEngine` instead - see the
next section.

## Multi-asset engine (S6)

`MultiAssetEngine` interleaves several symbols' tick streams by timestamp and
drives one `MultiAssetStrategy` across all of them. Implement `on_tick` with
a `symbol` argument instead of the single-symbol `Strategy` protocol:

```python
from mmbt.core.protocol import BaseMultiAssetStrategy
from mmbt.engine.multi_asset import MultiAssetEngine

class CrossAssetMM(BaseMultiAssetStrategy):
    def on_tick(self, symbol: str, book: OrderBook, trades: list[Trade]) -> list[Order | CancelOrder]:
        # symbol tells you which stream this tick came from -- react across
        # assets in one place instead of running N independent engines
        ...

engine = MultiAssetEngine(CrossAssetMM(), fee_rate_maker=0.0001, fee_rate_taker=0.0005)
results = engine.run({
    "BTC-USD": btc_ticks,   # each an Iterable[MarketTick]
    "ETH-USD": eth_ticks,
})
print(results["BTC-USD"].summary(), results["ETH-USD"].summary())
```

Each symbol still gets independent inventory/PnL and its own resting-order
book - only the tick ordering and the strategy driving loop are shared. Fill
model matches `BacktestEngine` (passive heuristic + taker/reject on crossing
orders, no FIFO queue or latency simulation yet); there's no "Pro" multi-asset
engine yet.

## Exchange adapters (S4)

Wrap CSV/Parquet data in a `CSVExchange` to match the `Exchange` protocol:

```python
from mmbt.data.exchange import CSVExchange, ExchangeMetadata

ex = CSVExchange(name="hyperliquid")
ex.register("BTC-USD", "data/btc_ticks.csv",
            ExchangeMetadata(tick_size=0.1, fee_rate_maker=0.0001, fee_rate_taker=0.0005))

for tick in ex.load_ticks("BTC-USD", start_ts=0, end_ts=1e13):
    ...
```

Implement the `Exchange` protocol to add live feed or other venue adapters.
`fee_rate_maker`/`fee_rate_taker` here map straight through to
`ProBacktestEngine`/`BacktestEngine`'s constructor args of the same name -
pass a negative `fee_rate_maker` yourself if your venue pays rebates.

## Open interest (S6)

`OpenInterestSchedule` is a standalone, engine-agnostic lookup - not part of
the tick stream or any engine. OI typically updates far less often than the
book (seconds to minutes, not ticks), so it isn't baked into
`MarketTick`/`OrderBook`; build one yourself and query it from inside your
own strategy using the current `book.ts`:

```python
from mmbt.data import OpenInterestSchedule

oi_sched = OpenInterestSchedule.from_csv("btc_oi.csv")  # columns: ts, oi

class OiAwareMM(BaseStrategy):
    def __init__(self, oi_sched: OpenInterestSchedule):
        self.oi_sched = oi_sched

    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list[Order | CancelOrder]:
        oi        = self.oi_sched.as_of(book.ts)             # forward-filled, None before first point
        oi_change = self.oi_sched.change(book.ts, lookback_us=60_000_000.0)  # last minute's delta
        ...
```

No engine or protocol changes needed - the schedule is just data your
strategy happens to hold a reference to. `OpenInterestSchedule.from_dict(...)`
and the synthetic `generate_oi_schedule(...)` (random walk, same spirit as
`data/synthetic.py`) are also available for testing without real OI data.


## Inventory skew strategy (S4)

`InventorySkewMM` is the second reference implementation - shifts quotes based on position:

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

Define a module-level `run_fn` (no lambdas - they can't be pickled by `ProcessPoolExecutor`):

```python
# examples/sweep_example.py: run_fn at module level
from mmbt.core.types import MarketTick
from mmbt.reporting.metrics import StrategyMetrics
from mmbt.engine.pro import ProBacktestEngine
from examples.strategies.symmetric_mm import SymmetricMM

def run_symmetric_mm(params: dict, ticks: list[MarketTick]) -> StrategyMetrics:
    engine = ProBacktestEngine(fee_rate_maker=0.0001, fee_rate_taker=0.0005)
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

### Out-of-sample validation (S7)

A grid search finds whatever config performed best on the data you gave it —
that's not the same as finding a real edge. `out_of_sample_validate` splits
`ticks` chronologically (no shuffling, this is time-series data), sweeps the
grid on the training slice, then re-runs only the top candidates on the
held-out slice so you can see whether the in-sample winner actually holds up:

```python
from mmbt.engine.sweep import out_of_sample_validate

results = out_of_sample_validate(grid, run_fn=run_symmetric_mm, ticks=ticks,
                                  train_frac=0.8, top_n=5)
for r in results:
    print(r.params, r.in_sample_pnl, r.out_of_sample_pnl, r.degradation, r.overfit_flag)
```

`overfit_flag` is a cheap heuristic (profitable in-sample, unprofitable
out-of-sample) — not a rigorous test. Use `degradation` and your own judgement
too.

### Walk-forward testing (S7)

A single train/test split can land in one lucky (or unlucky) regime.
`walk_forward_validate` rolls the split forward through the data in several
folds, re-optimizing and re-validating each time — more relevant for
microstructure than a static split, since vol/spread/flow regimes shift
within a single day:

```python
from mmbt.engine.sweep import walk_forward_validate, walk_forward_summary

folds   = walk_forward_validate(grid, run_fn=run_symmetric_mm, ticks=ticks,
                                 n_folds=5, expanding=False)
summary = walk_forward_summary(folds)
print(summary)  # oos_profitable_frac, avg/std OOS net_pnl, winning-config consistency
```

`expanding=False` (default) uses a fixed-size rolling training window (recent
history only); `expanding=True` grows the window from the start each fold
(all history seen so far). Low `param_consistency` in the summary — a
different config wins every fold — is itself a signal: that's a coin flip,
not a strategy with a stable edge.

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
plots.equity_curve(report)               # cumulative PnL, realized vs total
plots.inventory_over_time(report)        # position over time
plots.adverse_selection(report)          # are we getting picked off?
plots.fill_analysis(report)              # queue depth at fill (FIFO diagnostic)
plots.fill_latency_distribution(report)  # observed order/cancel latency vs the lognormal model
                                          # (ProBacktestEngine only -- BacktestEngine/MultiAssetEngine
                                          # have no latency simulation to plot)
```

### Comparing multiple runs (S6)

`param_heatmap`/`ranking_table` (see Parameter sweeps below) tell you which
config won; `equity_curves_overlay` shows *how* -- overlay several sweep
runs' equity curves to see drawdown timing or edge decay that a single
net_pnl number hides:

```python
from mmbt.reporting.sweep_plots import equity_curves_overlay

equity_curves_overlay(results, top_n=5, by="net_pnl")  # best 5 configs, superimposed
```

### Interactive HTML export (S6)

The plots above are static matplotlib. For a zoomable/pannable dashboard you
can share as a single file:

```python
from mmbt.reporting.html_export import export_html_dashboard

export_html_dashboard(report, "dashboard.html", title="BTC-USD symmetric MM")
```

Self-contained by default (`include_plotlyjs=True` embeds plotly.js, ~3-4MB,
works fully offline). Pass `include_plotlyjs="cdn"` for a much smaller file
that needs internet to view. Panels with no data (e.g. latency on a
`BacktestEngine` run) render as a labeled blank instead of erroring.

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

CSV format (trade columns optional - leave empty for book-only ticks):
```
ts,bid_px,bid_sz,ask_px,ask_sz,trade_px,trade_sz,trade_side
1000.0,49990.0,1.5,50000.0,2.0,,,
2000.0,49985.0,1.2,49995.0,1.8,49985.0,0.5,SELL
```

Multi-level book (optional, 5-10 levels recommended for realistic queue simulation):
add `bid_px_2,bid_sz_2,ask_px_2,ask_sz_2`, `bid_px_3,bid_sz_3,...` and so on -
level 1 stays bare (`bid_px`/`bid_sz`) for backward compatibility. Levels are
read in order and stop at the first missing/blank one, so a thinner book on
some rows (exchange only sent N levels that tick) is fine, not an error:
```
ts,bid_px,bid_sz,ask_px,ask_sz,bid_px_2,bid_sz_2,ask_px_2,ask_sz_2
1000.0,49990.0,1.5,50000.0,2.0,49985.0,3.0,50005.0,2.5
```
Same convention for Parquet.

## Architecture

```
mmbt/
├── core/
│   ├── types.py: BookLevel, OrderBook, Trade, Order, CancelOrder, Fill, InventoryState
│   ├── protocol.py: Strategy, MultiAssetStrategy, RiskManager, Exchange (typing.Protocol)
│   └── portfolio.py: Portfolio (cross-symbol PnL/position aggregation)
│
├── data/
│   ├── loader.py: TickLoader (CSV chunked, Parquet row-group streamed, multi-level book)
│   ├── synthetic.py: SyntheticConfig, generate_ticks (random-walk smoke-test data)
│   ├── exchange.py: CSVExchange, ExchangeMetadata (fee_rate_maker/taker, tick_size, ...)
│   └── open_interest.py: OpenInterestSchedule (standalone OI lookup, not engine-coupled)
│
├── queue/
│   ├── fifo.py: FIFOQueueState, FIFOQueueSimulator (iceberg detection, cancel support,
│   │                        Rust-accelerated when available)
│   ├── cancel_models.py: ReduceRatioCancelModel, ProbQueueCancelModel
│   ├── passive.py: PassiveFillSimulator + try_fill_orders (used by BacktestEngine /
│   │                        MultiAssetEngine, partial-fill-aware order sweep)
│   └── taker.py: crosses_book, sweep_book (taker/IOC execution against visible depth)
│
├── latency/
│   ├── config.py: LatencyConfig (Pydantic, YAML/JSON serializable)
│   ├── simulator.py: LatencySimulator (min-heap event queue)
│   └── book_history.py: BookHistory (ring buffer feeding the stale-book feed delay)
│
├── risk/
│   └── base.py: NullRiskManager, MaxInventoryRiskManager
│
├── reporting/
│   ├── metrics.py: StrategyMetrics, BacktestReport, EquitySnapshot, FillRecord
│   ├── mid_history.py: MidHistoryBuffer (fixed-capacity numpy circular buffer)
│   ├── plots.py: matplotlib, equity curve, inventory, adverse selection, fill analysis, fill latency
│   ├── sweep_plots.py: matplotlib, param heatmap, PnL vs adverse selection, ranking table, equity overlay
│   └── html_export.py: Plotly interactive HTML dashboard (single portable file)
│
└── engine/
    ├── simple.py: BacktestEngine (fast, heuristic fills, no latency)
    ├── pro.py: ProBacktestEngine (FIFO queue + latency, the realistic one)
    ├── multi_asset.py: MultiAssetEngine (interleaves multiple symbols by timestamp)
    └── sweep.py: ParameterSweep, expand_grid (parallel grid search)
```

---

## Cancel models

The cancel model controls how much of the queue ahead of you evaporates on each aggressive trade. Two implementations ship out of the box, both Rust-accelerated:

| Model | Behaviour | Good for |
|---|---|---|
| `ReduceRatioCancelModel(0.20)` | Fixed 20% of in-front queue cancels per trade | Fast, interpretable default |
| `ProbQueueCancelModel(min=0.05, max=0.70)` | Cancel rate scales with `trade_size / qty_in_front` | More realistic on deep books |

Implement `CancelModel` protocol to plug in your own calibration - a custom model falls back to the pure-Python queue path automatically (Rust can't call back into arbitrary Python cancellation logic).


---

## Latency config

```yaml
# examples/configs/pro_backtest.yaml
feed_us: 100.0      # feed-to-strategy latency (median, microseconds)
order_us: 450.0     # order round-trip (median, microseconds)
cancel_us: 280.0    # cancel round-trip (median, microseconds)
jitter: 0.20        # lognormal sigma - 0.20 fits most co-lo setups
```

Latencies are sampled from a lognormal distribution per-event. Increase `jitter` to model noisier network paths.

---

## Roadmap

| Sprint | Status | Scope |
|---|---|---|
| S1: Foundation | done | Core types, FIFO queue, cancel support, both engines |
| S2: Reporting | done | Equity curve, inventory heatmap, adverse selection charts, TickLoader |
| S3: Engine + Sweep | done | ParameterSweep, parallel runs, sweep plots |
| S4: Launch | done | Exchange Protocol, Portfolio, InventorySkewMM, 80 tests |
| S5: Rust hot path | done | PyO3 port of FIFOQueueSimulator for multi-month datasets |
| S6: Engine realism | done | Taker orders (fill or reject on crossing), real `queue_displacement_us`, `MultiAssetEngine`, maker/taker fee split |
| S7: Reporting + sweep validation | done | `fill_latency_distribution`, `equity_curves_overlay`, interactive Plotly HTML export, `OpenInterestSchedule`, `out_of_sample_validate`, `walk_forward_validate` |

No CI pipeline is wired up yet (`pytest tests/` locally is the only gate right now) - on the roadmap, see Known limitations.

---

## Known limitations

- **`MultiAssetEngine` has no FIFO/latency-aware counterpart yet** - it matches `BacktestEngine`'s fill model (passive heuristic + taker/reject), not `ProBacktestEngine`'s. Use `Portfolio` + N `ProBacktestEngine`s if you need FIFO realism across multiple symbols and don't need them to react to each other within a tick.
- **CSV/Parquet book depth** is capped by whatever your data provides the loader reads as many levels as you give it, but doesn't synthesize missing ones.
- **No CI** - tests only run when someone runs them locally.

---

## Contact

Built by **Taha** - algorithmic trader focused on execution and microstructure.

- Email: fadilrezokt@gmail.com
