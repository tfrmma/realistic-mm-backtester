from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mmbt.core.protocol import RiskManager, Strategy
from mmbt.core.types import CancelOrder, Fill, InventoryState, MarketTick, Order, OrderBook
from mmbt.latency.book_history import BookHistory
from mmbt.latency.config import LatencyConfig
from mmbt.latency.simulator import LatencySimulator
from mmbt.queue.cancel_models import CancelModel, ReduceRatioCancelModel
from mmbt.queue.fifo import FIFOQueueSimulator
from mmbt.queue.taker import crosses_book, sweep_book
from mmbt.reporting.metrics import EquitySnapshot, FillRecord, StrategyMetrics
from mmbt.risk.base import NullRiskManager


@dataclass
class _StratState:
    strategy: Strategy
    inventory: InventoryState
    latency_sim: LatencySimulator
    queue_sim: FIFOQueueSimulator
    metrics: StrategyMetrics
    book_history: BookHistory
    snapshot_every: int = 100
    # order_id -> ts when the order actually landed in the queue (for
    # queue_displacement_us). order_id -> cancel's expected arrival ts, while
    # that cancel is still in flight (removed once it lands or the order fills)
    order_register_ts: dict[str, float] = field(default_factory=dict, repr=False)
    pending_cancels: dict[str, float] = field(default_factory=dict, repr=False)
    _tick_count: int = field(default=0, repr=False, init=False)


class ProBacktestEngine:
    """
    FIFO queue + latency simulation. This is the one you trust.

    Tick flow:
      1. Land pending orders/cancels (latency sim). A landing order that
         crosses the true book executes immediately as a taker fill (or gets
         rejected if is_post_only) instead of joining the FIFO queue.
      2. FIFO queue processes tick -> maker fills (against the TRUE current
         book — fills always reflect what actually happened on the exchange)
      3. Strategy on_tick, fed a book/trades snapshot delayed by a sampled
         feed_us (ring buffer of recent ticks, see latency/book_history.py) —
         the strategy decides based on what it *would* have seen, not the
         current tick
      4. Risk check -> latency sim for next arrival
      5. Equity snapshot every snapshot_every ticks

    Pass seed for reproducible runs.
    """

    def __init__(
        self,
        latency_config: LatencyConfig | None = None,
        cancel_model: CancelModel | None = None,
        risk: RiskManager | None = None,
        fee_rate_maker: float = 0.0,
        fee_rate_taker: float = 0.0,
        snapshot_every: int = 100,
        seed: int | None = None,
        book_history_size: int = 2_000,
        mid_history_capacity: int = 200_000,
    ) -> None:
        self._lat_cfg          = latency_config or LatencyConfig()
        self._cancel_model     = cancel_model or ReduceRatioCancelModel()
        self._risk             = risk or NullRiskManager()
        self._fee_rate_maker   = fee_rate_maker
        self._fee_rate_taker   = fee_rate_taker
        self._snapshot_every   = snapshot_every
        self._seed             = seed
        self._book_history_size    = book_history_size
        self._mid_history_capacity = mid_history_capacity
        self._strats: dict[str, _StratState] = {}

    def add_strategy(self, name: str, strategy: Strategy, symbol: str) -> None:
        if name in self._strats:
            raise ValueError(f"strategy '{name}' already registered")
        # derive per-strategy seed so multi-strategy runs are still deterministic
        strat_seed = None if self._seed is None else self._seed + hash(name) % (2 ** 32)
        self._strats[name] = _StratState(
            strategy=strategy,
            inventory=InventoryState(symbol=symbol),
            latency_sim=LatencySimulator(self._lat_cfg, seed=strat_seed),
            queue_sim=FIFOQueueSimulator(self._cancel_model),
            metrics=StrategyMetrics(symbol=symbol, mid_history_capacity=self._mid_history_capacity),
            book_history=BookHistory(maxlen=self._book_history_size),
            snapshot_every=self._snapshot_every,
        )

    def run(self, ticks: Iterable[MarketTick]) -> dict[str, StrategyMetrics]:
        for tick in ticks:
            for state in self._strats.values():
                self._step(state, tick)
        return {name: s.metrics for name, s in self._strats.items()}

    def _step(self, state: _StratState, tick: MarketTick) -> None:
        ts     = tick.ts
        book   = tick.book
        trades = tick.trades

        self._land_pending(state, ts, book)

        for fill in state.queue_sim.process_tick(book, trades):
            if fill.order_id in state.pending_cancels:
                # a cancel was in flight for this order but the fill beat it
                # to the exchange, how much queue time had the order already
                # accumulated by the time our (too-late) cancel would have landed
                register_ts = state.order_register_ts.get(fill.order_id, fill.ts)
                fill.queue_displacement_us = state.pending_cancels[fill.order_id] - register_ts
                del state.pending_cancels[fill.order_id]
            state.order_register_ts.pop(fill.order_id, None)
            self._record_fill(state, fill, book)

        state.metrics.mid_history.append((ts, book.mid))

        state._tick_count += 1
        if state._tick_count % state.snapshot_every == 0:
            state.metrics.equity_snapshots.append(EquitySnapshot(
                ts=ts,
                realized_pnl=state.inventory.realized_pnl,
                unrealized_pnl=state.inventory.unrealized_pnl(book.mid),
                position=state.inventory.position,
                fees_paid=state.inventory.fees_paid,
            ))

        # true tick goes into the ring buffer first, then the strategy is fed
        # whatever it would actually have received feed_us later, fills above
        # already happened against the TRUE book, only the strategy's view is stale
        state.book_history.push(ts, book, trades)
        feed_delay   = state.latency_sim.feed_delay_us()
        seen         = state.book_history.as_of(ts - feed_delay)
        assert seen is not None  # we just pushed, buffer can't be empty
        actions      = state.strategy.on_tick(seen.book, seen.trades)
        new_orders = self._risk.check(
            [a for a in actions if isinstance(a, Order)],
            state.inventory, book,
        )
        for order in new_orders:
            order.ts = ts
            state.latency_sim.submit_order(order, ts)
        for cancel in [a for a in actions if isinstance(a, CancelOrder)]:
            arrive_ts = state.latency_sim.submit_cancel(cancel, ts)
            state.pending_cancels[cancel.order_id] = arrive_ts

    def _land_pending(self, state: _StratState, ts: float, book: OrderBook) -> None:
        for kind, payload in state.latency_sim.poll_arrived(ts):
            if kind == "order":
                self._land_order(state, payload, ts, book)
            elif kind == "cancel":
                state.queue_sim.cancel(payload.order_id)
                state.pending_cancels.pop(payload.order_id, None)
                state.order_register_ts.pop(payload.order_id, None)

    def _land_order(self, state: _StratState, order: Order, ts: float, book: OrderBook) -> None:
        # evaluated against the TRUE book at landing time, not whatever the
        # strategy saw when it decided to submit, latency can turn a resting
        # order into a crossing one (or vice versa) by the time it arrives
        if not crosses_book(order, book):
            state.queue_sim.register(order, book)
            state.order_register_ts[order.order_id] = ts
            return
        if order.is_post_only:
            state.metrics.rejected_orders += 1
            return
        execution = sweep_book(order, book)
        if execution is None:
            return  # crossed on paper but the crossed side had no real depth
        self._record_fill(state, Fill(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            price=execution.vwap_price, size=execution.filled_size,
            is_maker=False, ts=ts,
        ), book)
        # any unfilled remainder does NOT rest IOC semantics, like a real taker order

    def _record_fill(self, state: _StratState, fill: Fill, book: OrderBook) -> None:
        state.inventory.apply_fill(fill, self._fee_rate_maker, self._fee_rate_taker)
        state.strategy.on_fill(fill)
        state.metrics.fills.append(fill)
        state.metrics.fill_records.append(FillRecord(fill=fill, mid_at_fill=book.mid))
        state.metrics.realized_pnl = state.inventory.realized_pnl
        state.metrics.fees_paid    = state.inventory.fees_paid
