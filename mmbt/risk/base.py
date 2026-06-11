from __future__ import annotations

from mmbt.core.types import InventoryState, Order, OrderBook, Side


class NullRiskManager:
    """Passes everything through. Replace before going live."""

    def check(self, orders: list[Order], inventory: InventoryState, book: OrderBook) -> list[Order]:
        return orders


class MaxInventoryRiskManager:
    """
    Blocks orders that would push |position| past max_position.
    Circuit breaker, not a real risk system.
    TODO: add per-symbol notional limits when multi-asset engine lands.
    """

    def __init__(self, max_position: float) -> None:
        if max_position <= 0:
            raise ValueError("max_position must be positive")
        self.max_position = max_position

    def check(self, orders: list[Order], inventory: InventoryState, book: OrderBook) -> list[Order]:
        allowed = []
        for o in orders:
            delta = o.size if o.side == Side.BUY else -o.size
            if abs(inventory.position + delta) <= self.max_position:
                allowed.append(o)
        return allowed
