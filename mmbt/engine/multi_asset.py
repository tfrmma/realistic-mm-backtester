from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from mmbt.core.protocol import MultiAssetStrategy, RiskManager
from mmbt.core.types import CancelOrder, Fill, InventoryState, MarketTick, Order, OrderBook
from mmbt.queue.passive import PassiveFillSimulator, try_fill_orders
from mmbt.queue.taker import crosses_book, sweep_book
from mmbt.reporting.metrics import EquitySnapshot, FillRecord, StrategyMetrics
from mmbt.risk.base import NullRiskManager


@dataclass
class _SymbolState:
    inventory: InventoryState
    metrics: StrategyMetrics
    pending: list[Order] = field(default_factory=list)
    _tick_count: int = field(default=0, repr=False, init=False)


class MultiAssetEngine:
    """
    Interleaves tick streams from multiple symbols by timestamp and drives a
    single MultiAssetStrategy instance across all of them, the strategy
    finds out which symbol each tick belongs to and can react across symbols
    in one decision (cross-asset hedging, correlated quoting, portfolio-level
    risk), instead of running N independent single-symbol engines that can't
    see each other at all.

    Fill model matches BacktestEngine: passive heuristic fills plus immediate
    taker execution / post-only rejection for orders that cross the book (see
    queue/taker.py), no latency or FIFO queue simulation. There's no "Pro"
    (FIFO + latency) multi-asset counterpart yet, each symbol gets its own
    independent inventory and resting-order book, only the tick ordering and
    the strategy driving loop are shared across symbols.
    """

    def __init__(
        self,
        strategy: MultiAssetStrategy,
        fill_sim: PassiveFillSimulator | None = None,
        risk: RiskManager | None = None,
        fee_rate_maker: float = 0.0,
        fee_rate_taker: float = 0.0,
        snapshot_every: int = 100,
        mid_history_capacity: int = 200_000,
    ) -> None:
        self._strategy        = strategy
        self._fill_sim         = fill_sim or PassiveFillSimulator()
        self._risk             = risk or NullRiskManager()
        self._fee_rate_maker   = fee_rate_maker
        self._fee_rate_taker   = fee_rate_taker
        self._snapshot_every   = snapshot_every
        self._mid_history_capacity = mid_history_capacity
        self._symbols: dict[str, _SymbolState] = {}

    def _state(self, symbol: str) -> _SymbolState:
        if symbol not in self._symbols:
            self._symbols[symbol] = _SymbolState(
                inventory=InventoryState(symbol=symbol),
                metrics=StrategyMetrics(symbol=symbol, mid_history_capacity=self._mid_history_capacity),
            )
        return self._symbols[symbol]

    def run(self, streams: dict[str, Iterable[MarketTick]]) -> dict[str, StrategyMetrics]:
        for symbol, tick in _merge_by_ts(streams):
            self._step(symbol, tick)
        return {symbol: st.metrics for symbol, st in self._symbols.items()}

    def _step(self, symbol: str, tick: MarketTick) -> None:
        state = self._state(symbol)
        book, trades, ts = tick.book, tick.trades, tick.ts

        fills, remaining = try_fill_orders(self._fill_sim, state.pending, trades, book, ts)
        state.pending = remaining
        for _, fill in fills:
            self._record_fill(state, fill, book)

        state.metrics.mid_history.append((ts, book.mid))

        state._tick_count += 1
        if state._tick_count % self._snapshot_every == 0:
            state.metrics.equity_snapshots.append(EquitySnapshot(
                ts=ts,
                realized_pnl=state.inventory.realized_pnl,
                unrealized_pnl=state.inventory.unrealized_pnl(book.mid),
                position=state.inventory.position,
                fees_paid=state.inventory.fees_paid,
            ))

        actions    = self._strategy.on_tick(symbol, book, trades)
        cancels    = {a.order_id for a in actions if isinstance(a, CancelOrder)}
        new_orders = [a for a in actions if isinstance(a, Order)]

        if cancels:
            state.pending = [o for o in state.pending if o.order_id not in cancels]

        for order in self._risk.check(new_orders, state.inventory, book):
            if not crosses_book(order, book):
                state.pending.append(order)
                continue
            if order.is_post_only:
                state.metrics.rejected_orders += 1
                continue
            execution = sweep_book(order, book)
            if execution is None:
                continue
            self._record_fill(state, Fill(
                order_id=order.order_id, symbol=order.symbol, side=order.side,
                price=execution.vwap_price, size=execution.filled_size,
                is_maker=False, ts=ts,
            ), book)

    def _record_fill(self, state: _SymbolState, fill: Fill, book: OrderBook) -> None:
        state.inventory.apply_fill(fill, self._fee_rate_maker, self._fee_rate_taker)
        self._strategy.on_fill(fill)
        state.metrics.fills.append(fill)
        state.metrics.fill_records.append(FillRecord(fill=fill, mid_at_fill=book.mid))
        state.metrics.realized_pnl = state.inventory.realized_pnl
        state.metrics.fees_paid    = state.inventory.fees_paid


def _merge_by_ts(streams: dict[str, Iterable[MarketTick]]) -> Iterator[tuple[str, MarketTick]]:
    """Chronological merge across per-symbol tick streams via a min-heap.
    A global counter breaks ties so heap entries never need to compare
    MarketTick objects directly (dataclass has no ordering defined)."""
    counter    = itertools.count()
    iterators  = {sym: iter(stream) for sym, stream in streams.items()}
    heap: list[tuple[float, int, str, MarketTick]] = []

    for sym, it in iterators.items():
        tick = next(it, None)
        if tick is not None:
            heapq.heappush(heap, (tick.ts, next(counter), sym, tick))

    while heap:
        ts, _, sym, tick = heapq.heappop(heap)
        yield sym, tick
        nxt = next(iterators[sym], None)
        if nxt is not None:
            heapq.heappush(heap, (nxt.ts, next(counter), sym, nxt))
