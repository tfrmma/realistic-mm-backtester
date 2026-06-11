"""Inventory-aware market maker. Skews quotes proportionally to position.

When long: shift both quotes down (discourage buys, attract sells).
When short: shift both quotes up (attract buys, discourage sells).
Skew = -inv_ratio * skew_bps * mid, where inv_ratio in [-1, 1].

Simplified Avellaneda-Stoikov. Good reference impl for inventory mgmt.
"""

from __future__ import annotations

import uuid

from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import CancelOrder, Fill, Order, OrderBook, Side, Trade


class InventorySkewMM(BaseStrategy):

    def __init__(
        self,
        symbol: str,
        half_spread_bps: float,
        order_size: float,
        max_position: float,
        skew_bps: float = 1.0,
    ) -> None:
        if max_position <= 0:
            raise ValueError("max_position must be positive")
        self.symbol       = symbol
        self.half_spread  = half_spread_bps / 10_000.0
        self.order_size   = order_size
        self.max_position = max_position
        self.skew         = skew_bps / 10_000.0
        self._position: float = 0.0
        self._bid_id: str | None = None
        self._ask_id: str | None = None

    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list[Order | CancelOrder]:
        actions: list[Order | CancelOrder] = []

        if self._bid_id:
            actions.append(CancelOrder(order_id=self._bid_id, symbol=self.symbol))
        if self._ask_id:
            actions.append(CancelOrder(order_id=self._ask_id, symbol=self.symbol))

        mid       = book.mid
        inv_ratio = max(-1.0, min(1.0, self._position / self.max_position))
        offset    = -inv_ratio * self.skew * mid

        bid_price = mid * (1.0 - self.half_spread) + offset
        ask_price = mid * (1.0 + self.half_spread) + offset

        if bid_price >= ask_price:
            return actions  # crossed book — skip this tick

        self._bid_id = str(uuid.uuid4())
        self._ask_id = str(uuid.uuid4())
        actions.append(Order(self._bid_id, self.symbol, Side.BUY,  bid_price, self.order_size, is_post_only=True))
        actions.append(Order(self._ask_id, self.symbol, Side.SELL, ask_price, self.order_size, is_post_only=True))
        return actions

    def on_fill(self, fill: Fill) -> None:
        self._position += fill.size if fill.side == Side.BUY else -fill.size
        if fill.order_id == self._bid_id:
            self._bid_id = None
        elif fill.order_id == self._ask_id:
            self._ask_id = None

    @property
    def position(self) -> float:
        return self._position
