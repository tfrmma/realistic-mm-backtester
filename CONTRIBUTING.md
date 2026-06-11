# Contributing

Contributions welcome. This is a focused repo — new features should serve
the core goal of realistic market making simulation.

## Setup

```bash
git clone https://github.com/tfrmma/realistic-mm-backtester
cd mmbt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python ≥ 3.11 required.

## Running tests

```bash
pytest tests/ -v
```

All tests run in under 30 seconds on a modern machine. If yours are slower,
check for accidental `n_jobs > 1` in sweep tests (those should use `n_jobs=1`).

## Code style

- `ruff check mmbt/` for linting
- Line length: 100 (configured in `pyproject.toml`)
- Type hints on all public functions
- Comments in English

## Adding an exchange adapter

Implement the `Exchange` protocol from `mmbt.core.protocol`:

```python
from mmbt.core.protocol import Exchange
from mmbt.core.types import MarketTick
from typing import Iterator

class MyExchange:
    name = "my_exchange"

    def load_ticks(self, symbol: str, start_ts: float, end_ts: float) -> Iterator[MarketTick]:
        ...

    def tick_size(self, symbol: str) -> float:
        ...

    def min_order_size(self, symbol: str) -> float:
        ...
```

No inheritance required — duck typing via `Protocol`. Drop a file in `mmbt/data/`
and open a PR.

## Adding a cancel model

Implement `CancelModel` from `mmbt.queue.cancel_models`:

```python
class MyModel:
    def cancelled_fraction(self, qty_in_front: float, trade_size: float) -> float:
        # return fraction of in-front queue that cancels alongside this trade
        ...
```

Add it to `mmbt/queue/cancel_models.py` with a calibration note.

## What belongs in this repo

- Queue simulation improvements (better iceberg detection, pro-rata fill models)
- New exchange adapters
- Reporting improvements (new plot types, HTML export)
- Performance (Rust hot path via PyO3 — see S5 roadmap)

## What doesn't belong here

- Live trading connectors
- Specific strategy implementations (beyond reference examples)
- Venue-specific alpha signals

## Contact

For questions about architecture or deployment: fadilrezokt@gmail.com
