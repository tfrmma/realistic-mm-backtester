from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mmbt.core.protocol import RiskManager, Strategy
from mmbt.core.types import CancelOrder, Fill, InventoryState, MarketTick, Order, OrderBook, Trade
from mmbt.queue.passive import PassiveFillSimulator
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
    Simple tick-by-tick engine. Fills are immediate when a trade crosses price.
    Optimistic, but fast good for parameter sweeps.
    Use ProBacktestEngine when fill realism matters.
    """

    def __init__(
        self,
        fill_sim: PassiveFillSimulator | None = None,
        risk: RiskManager | None = None,
        fee_rate: float = 0.0,
        snapshot_every: int = 100,
        mid_history_capacity: int = 200_000,
    ) -> None:
        self._fill_sim      = fill_sim or PassiveFillSimulator()
        self._risk          = risk or NullRiskManager()
        self._fee_rate      = fee_rate
        self._snapshot_every = snapshot_every
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
        fills, remaining = self._try_fill(state.pending, trades, book, ts)
        state.pending = remaining

        for _, fill in fills:
            state.inventory.apply_fill(fill, self._fee_rate)
            state.strategy.on_fill(fill)
            state.metrics.fills.append(fill)
            state.metrics.fill_records.append(FillRecord(fill=fill, mid_at_fill=book.mid))
            state.metrics.realized_pnl = state.inventory.realized_pnl
            state.metrics.fees_paid    = state.inventory.fees_paid

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

        actions   = state.strategy.on_tick(book, trades)
        cancels   = {a.order_id for a in actions if isinstance(a, CancelOrder)}
        new_orders = [a for a in actions if isinstance(a, Order)]

        if cancels:
            state.pending = [o for o in state.pending if o.order_id not in cancels]

        state.pending.extend(self._risk.check(new_orders, state.inventory, book))

    _DUST = 1e-9  # below this, an order can't produce a meaningful fill anymore

    def _try_fill(
        self,
        orders: list[Order],
        trades: list[Trade],
        book: OrderBook,
        ts: float,
    ) -> tuple[list[tuple[Order, Fill]], list[Order]]:
        # build (order, fill) pairs in the loop zip on misaligned lists was a previous bug
        #
        # PassiveFillSimulator.simulate() can return a fill smaller than
        # order.size (a partial fill, e.g. pro-rata share of a level's depth).
        # Only drop the order once it's fully exhausted -- the remainder keeps
        # resting at the same price/order_id and can catch further fills on
        # later ticks, same as a real passive order would.
        #
        # Eviction is based on the order's remaining size, not on whether this
        # tick produced a fill: PassiveFillSimulator has its own epsilon guard
        # (won't emit a fill sized <=1e-12), so a shrinking order can otherwise
        # get stuck forever just above that floor, never quite hitting it and
        # never getting swept out either. Checking size directly avoids that
        # dead zone.
        fills: list[tuple[Order, Fill]] = []
        remaining: list[Order] = []
        for order in orders:
            fill = self._fill_sim.simulate(order, trades, book)
            if fill:
                fill.ts = ts
                fills.append((order, fill))
                order.size -= fill.size
            if order.size > self._DUST:
                remaining.append(order)
        return fills, remaining
