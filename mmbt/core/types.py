from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple


class Side(IntEnum):
    BUY = 1
    SELL = -1  # IntEnum so it survives the FFI boundary


class BookLevel(NamedTuple):
    price: float
    size: float


@dataclass(slots=True)
class OrderBook:
    bids: list[BookLevel]
    asks: list[BookLevel]
    ts: float

    @property
    def mid(self) -> float:
        return (self.bids[0].price + self.asks[0].price) / 2.0

    @property
    def spread(self) -> float:
        return self.asks[0].price - self.bids[0].price

    @property
    def best_bid(self) -> BookLevel:
        return self.bids[0]

    @property
    def best_ask(self) -> BookLevel:
        return self.asks[0]


@dataclass(slots=True, frozen=True)
class Trade:
    price: float
    size: float
    side: Side
    ts: float
    is_liquidation: bool = False


@dataclass(slots=True)
class Order:
    order_id: str
    symbol: str
    side: Side
    price: float
    size: float
    is_post_only: bool = False
    ts: float = 0.0


@dataclass(slots=True, frozen=True)
class CancelOrder:
    order_id: str
    symbol: str


@dataclass(slots=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    price: float
    size: float
    is_maker: bool
    ts: float
    qty_in_front: float = 0.0
    queue_displacement_us: float = 0.0


@dataclass(slots=True)
class MarketTick:
    book: OrderBook
    trades: list[Trade]
    ts: float


OrderAction = Order | CancelOrder


@dataclass(slots=True)
class InventoryState:
    symbol: str
    position: float = 0.0
    avg_entry: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    def apply_fill(self, fill: Fill, fee_rate: float = 0.0) -> None:
        sign = 1.0 if fill.side == Side.BUY else -1.0
        notional = fill.price * fill.size

        if self.position == 0.0:
            self.avg_entry = fill.price
        elif sign * self.position > 0:
            total = abs(self.position) * self.avg_entry + fill.size * fill.price
            self.avg_entry = total / (abs(self.position) + fill.size)
        else:
            closing = min(fill.size, abs(self.position))
            self.realized_pnl += (fill.price - self.avg_entry) * closing * (-sign)
            if fill.size > abs(self.position):
                self.avg_entry = fill.price

        self.position += sign * fill.size
        fee = notional * fee_rate
        self.fees_paid += -fee if fill.is_maker else fee

    def unrealized_pnl(self, mid: float) -> float:
        if self.position == 0.0:
            return 0.0
        return (mid - self.avg_entry) * self.position

    def total_pnl(self, mid: float) -> float:
        return self.realized_pnl + self.unrealized_pnl(mid) - self.fees_paid
