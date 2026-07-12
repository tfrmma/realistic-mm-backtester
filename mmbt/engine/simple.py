from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mmbt.core.protocol import RiskManager, Strategy
from mmbt.core.types import CancelOrder, Fill, InventoryState, MarketTick, Order, OrderBook, Trade
from mmbt.queue.passive import PassiveFillSimulator, try_fill_orders
from mmbt.queue.taker import crosses_book, sweep_book
from mmbt.reporting.metrics import EquitySnapshot, FillRecord, StrategyMetrics
from mmbt.risk.base import NullRiskManager


@dataclass
class _StratState:
    strategy: Strategy
    inventory: InventoryState
    metrics: StrategyMetrics
    pending: list[Order] = field(default_factory=list)
    snapshot_every: int = 100
    _tick_count: int = field(default=0, repr=False, init=False)


class BacktestEngine:
    """
    Simple tick-by-tick engine. Passive fills are immediate when a trade
    crosses a resting order's price. Optimistic, but fast, good for
    parameter sweeps. Use ProBacktestEngine when fill realism matters.

    Orders that cross the book at submission (order.price already inside the
    spread) are handled as taker orders instead of resting: filled
    immediately against visible depth (see queue/taker.py), or rejected if
    is_post_only=True, same as a real exchange would reject a post-only
    order that would take liquidity. Any unfilled remainder of a taker order
    does not rest (IOC semantics), matching a real market/IOC order.
    """

    def __init__(
        self,
        fill_sim: PassiveFillSimulator | None = None,
        risk: RiskManager | None = None,
        fee_rate_maker: float = 0.0,
        fee_rate_taker: float = 0.0,
        snapshot_every: int = 100,
        mid_history_capacity: int = 200_000,
    ) -> None:
        self._fill_sim         = fill_sim or PassiveFillSimulator()
        self._risk             = risk or NullRiskManager()
        self._fee_rate_maker   = fee_rate_maker
        self._fee_rate_taker   = fee_rate_taker
        self._snapshot_every   = snapshot_every
        self._mid_history_capacity = mid_history_capacity
        self._strats: dict[str, _StratState] = {}

    def add_strategy(self, name: str, strategy: Strategy, symbol: str) -> None:
        if name in self._strats:
            raise ValueError(f"strategy '{name}' already registered")
        self._strats[name] = _StratState(
            strategy=strategy,
            inventory=InventoryState(symbol=symbol),
            metrics=StrategyMetrics(symbol=symbol, mid_history_capacity=self._mid_history_capacity),
            snapshot_every=self._snapshot_every,
        )

    def run(self, ticks: Iterable[MarketTick]) -> dict[str, StrategyMetrics]:
        for tick in ticks:
            for state in self._strats.values():
                self._step(state, tick.book, tick.trades, tick.ts)
        return {name: s.metrics for name, s in self._strats.items()}

    def _step(self, state: _StratState, book: OrderBook, trades: list[Trade], ts: float) -> None:
        fills, remaining = try_fill_orders(self._fill_sim, state.pending, trades, book, ts)
        state.pending = remaining
        for _, fill in fills:
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

        actions    = state.strategy.on_tick(book, trades)
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
                continue  # crossed on paper but the crossed side had no real depth
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
