# L3 data capture

Standalone tools that get real L3 (order-by-order) market data into `mmbt`
without paying for a historical data vendor. Institutional L3 providers
(Tardis, CoinAPI Enterprise, Kaiko) run hundreds to low-thousands of dollars
a month; these scripts capture the same granularity for free, straight from
exchange-public WebSocket feeds, at the cost of only having data from
whenever you start recording.

Not part of the `mmbt` package these are CLI tools you run directly, not
modules you import (the loader that *does* get imported into your backtests
lives in `mmbt/data/l3_bitfinex.py`).

## Install

```bash
pip install -e ".[capture]"
```

## Workflow

**1. Capture** leave this running (background / VPS) to accumulate history:

```bash
python scripts/bitfinex_l3_listener.py --symbols tBTCUSD --output-dir ./l3
```

Subscribes to Bitfinex's public Raw Book (`R0`, true L3) and `trades`
channels, no API key needed. Writes batched `.parquet` files as it goes
see the file's own docstring for the full message-format notes and known
limitations (only Bitfinex and Coinbase expose L3 publicly; other exchanges
don't offer it at any price for retail).

**2. Merge** once you've got a batch of files, consolidate them:

```bash
python scripts/merge_l3.py ./l3 --output ./l3_merged/tBTCUSD.parquet --symbol tBTCUSD
```

Optional `mmbt.data.l3_bitfinex.BitfinexL3Exchange` reads a directory of
many small files just fine. Merging is for convenience (one file to move
around) and to drop exact-duplicate rows if a capture session overlapped.

**3. Backtest** feed it into the engine like any other Exchange:

```python
from mmbt.data.l3_bitfinex import BitfinexL3Exchange
from mmbt.data.exchange import ExchangeMetadata
from mmbt.engine.pro import ProBacktestEngine
from examples.strategies.symmetric_mm import SymmetricMM

ex = BitfinexL3Exchange()
ex.register("tBTCUSD", "./l3_merged", ExchangeMetadata(tick_size=0.1))

engine = ProBacktestEngine(fee_rate_maker=0.0001, fee_rate_taker=0.0005)
engine.add_strategy("mm", SymmetricMM("tBTCUSD", half_spread_bps=2.0, order_size=0.01), "tBTCUSD")
results = engine.run(list(ex.load_ticks("tBTCUSD")))
print(results["mm"].summary())
```

## Known limitations

- Only forward-looking: no retroactive history, only what you capture.
- Only Bitfinex exposes true L3 publicly Binance and most other
  major venues cap out at L2, at any price tier.
- The listener needs a stable, long-running process (or reconnect handling
  will do its job more often than you'd like) a small VPS works better
  than a laptop that sleeps.
