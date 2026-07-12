from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from mmbt.core.types import Fill, InventoryState, Order, OrderAction, OrderBook, Trade


@runtime_checkable
class Strategy(Protocol):
    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list[OrderAction]: ...
    def on_fill(self, fill: Fill) -> None: ...


@runtime_checkable
class MultiAssetStrategy(Protocol):
    """
    Like Strategy, but on_tick also gets told which symbol this tick belongs
    to needed by MultiAssetEngine, which interleaves several symbols'
    tick streams and drives a single strategy instance across all of them.
    Kept as a separate protocol (not an extension of Strategy) so existing
    single-symbol strategies and BacktestEngine/ProBacktestEngine are
    completely unaffected.
    """
    def on_tick(self, symbol: str, book: OrderBook, trades: list[Trade]) -> list[OrderAction]: ...
    def on_fill(self, fill: Fill) -> None: ...


@runtime_checkable
class RiskManager(Protocol):
    def check(self, orders: list[Order], inventory: InventoryState, book: OrderBook) -> list[Order]: ...


@runtime_checkable
class Exchange(Protocol):
    name: str
    def load_ticks(self, symbol: str, start_ts: float, end_ts: float) -> Iterator: ...
    def tick_size(self, symbol: str) -> float: ...
    def min_order_size(self, symbol: str) -> float: ...


class BaseStrategy:
    """Optional base, override what you need, ignore what you don't."""
    def on_fill(self, fill: Fill) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_end(self) -> None:
        pass


class BaseMultiAssetStrategy:
    """Optional base for MultiAssetStrategy, same idea as BaseStrategy."""
    def on_fill(self, fill: Fill) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_end(self) -> None:
        pass
