"""Simple symmetric market maker. Posts bid/ask around mid, re-quotes every tick.
Reference implementation — not something you'd actually trade as-is."""

from __future__ import annotations

import uuid

from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import CancelOrder, Fill, Order, OrderBook, Side, Trade


class SymmetricMM(BaseStrategy):
    """
    Posts one bid and one ask at mid +/- half_spread_bps. Re-quotes every tick.
    No inventory management — flat-out symmetric.
    """

    def __init__(self, symbol: str, half_spread_bps: float, order_size: float) -> None:
        self.symbol      = symbol
        self.half_spread = half_spread_bps / 10_000.0
        self.order_size  = order_size
        self._bid_id: str | None = None
        self._ask_id: str | None = None

    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list[Order | CancelOrder]:
        actions: list[Order | CancelOrder] = []

        if self._bid_id:
            actions.append(CancelOrder(order_id=self._bid_id, symbol=self.symbol))
        if self._ask_id:
            actions.append(CancelOrder(order_id=self._ask_id, symbol=self.symbol))

        mid = book.mid
        self._bid_id = str(uuid.uuid4())
        self._ask_id = str(uuid.uuid4())

        actions.append(Order(
            order_id=self._bid_id, symbol=self.symbol,
            side=Side.BUY, price=mid * (1.0 - self.half_spread),
            size=self.order_size, is_post_only=True,
        ))
        actions.append(Order(
            order_id=self._ask_id, symbol=self.symbol,
            side=Side.SELL, price=mid * (1.0 + self.half_spread),
            size=self.order_size, is_post_only=True,
        ))
        return actions

    def on_fill(self, fill: Fill) -> None:
        if fill.order_id == self._bid_id:
            self._bid_id = None
        elif fill.order_id == self._ask_id:
            self._ask_id = None
