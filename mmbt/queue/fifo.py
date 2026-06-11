from __future__ import annotations

from dataclasses import dataclass, field

from mmbt.core.types import Fill, Order, OrderBook, Side, Trade
from mmbt.queue.cancel_models import CancelModel, ReduceRatioCancelModel

try:
    from mmbt._core import FIFOQueueCore as _RustCore
    RUST_AVAILABLE = True
except ImportError:
    _RustCore = None       # type: ignore[assignment,misc]
    RUST_AVAILABLE = False


@dataclass
class FIFOQueueState:
    order: Order
    qty_in_front: float
    _cancelled: bool = field(default=False, repr=False)
    _filled:    bool = field(default=False, repr=False)

    @property
    def is_active(self) -> bool:
        return not self._cancelled and not self._filled

    def cancel(self) -> bool:
        if not self.is_active:
            return False
        self._cancelled = True
        return True

    def process_trade(self, trade: Trade, cancel_model: CancelModel) -> Fill | None:
        # don't touch this logic without running queue regression tests first
        if not self.is_active:
            return None
        order = self.order
        if not _trade_matches(trade, order):
            return None

        if self.qty_in_front > 0:
            frac     = cancel_model.cancelled_fraction(self.qty_in_front, trade.size)
            consumed = trade.size * (1.0 + frac)
            raw_q    = self.qty_in_front - consumed
            overshoot = max(0.0, -raw_q)
            self.qty_in_front = max(0.0, raw_q)
            if overshoot <= 0:
                return None
            fill_size = min(order.size, overshoot)
        else:
            fill_size = min(order.size, trade.size)

        if fill_size <= 1e-12:
            return None

        self._filled = True
        return Fill(
            order_id=order.order_id, symbol=order.symbol,
            side=order.side, price=order.price,
            size=fill_size, is_maker=True, ts=trade.ts,
            qty_in_front=self.qty_in_front,
        )


class FIFOQueueSimulator:
    """
    Manages queue positions for all resting orders.
    Uses Rust extension when available, pure Python otherwise.
    Check .using_rust to see which path is active.
    Build extension: maturin develop --release
    """

    def __init__(
        self,
        cancel_model: CancelModel | None = None,
        use_rust: bool = True,
    ) -> None:
        self._cancel_model = cancel_model or ReduceRatioCancelModel()

        if use_rust and RUST_AVAILABLE:
            cr = getattr(cancel_model, 'cancel_ratio', 0.20)
            self._core: object | None = _RustCore(cancel_ratio=cr)
            self._order_cache: dict[str, Order] = {}
        else:
            self._core = None
            self._states: dict[str, FIFOQueueState] = {}
            self._prev_book: dict[float, float] = {}

    @property
    def using_rust(self) -> bool:
        return self._core is not None

    def register(self, order: Order, book: OrderBook) -> None:
        qty = _qty_in_front(order, book)
        if self._core is not None:
            self._core.register(  # type: ignore[union-attr]
                order.order_id, order.symbol,
                int(order.side.value),
                order.price, order.size, order.ts, qty,
            )
            self._order_cache[order.order_id] = order
        else:
            self._states[order.order_id] = FIFOQueueState(order=order, qty_in_front=qty)

    def cancel(self, order_id: str) -> bool:
        if self._core is not None:
            self._order_cache.pop(order_id, None)
            return self._core.cancel(order_id)  # type: ignore[union-attr]
        s = self._states.get(order_id)
        if s is None:
            return False
        if s.cancel():
            del self._states[order_id]
            return True
        return False

    def process_tick(self, book: OrderBook, trades: list[Trade]) -> list[Fill]:
        if self._core is not None:
            return self._process_tick_rust(book, trades)
        return self._process_tick_python(book, trades)

    def active_orders(self) -> list[Order]:
        if self._core is not None:
            active = set(self._core.active_order_ids())  # type: ignore[union-attr]
            return [o for oid, o in self._order_cache.items() if oid in active]
        return [s.order for s in self._states.values() if s.is_active]

    def pending_count(self) -> int:
        if self._core is not None:
            return self._core.pending_count()  # type: ignore[union-attr]
        return len(self._states)

    def _process_tick_rust(self, book: OrderBook, trades: list[Trade]) -> list[Fill]:
        bids  = [(lvl.price, lvl.size) for lvl in book.bids]
        asks  = [(lvl.price, lvl.size) for lvl in book.asks]
        traw  = [(t.price, t.size, int(t.side.value), t.ts) for t in trades]
        fills = []
        for oid, sym, side_int, price, size, is_maker, ts, qif, qd in \
                self._core.process_tick(bids, asks, traw):  # type: ignore[union-attr]
            fills.append(Fill(
                order_id=oid, symbol=sym,
                side=Side(side_int), price=price, size=size,
                is_maker=is_maker, ts=ts,
                qty_in_front=qif, queue_displacement_us=qd,
            ))
            self._order_cache.pop(oid, None)
        return fills

    def _process_tick_python(self, book: OrderBook, trades: list[Trade]) -> list[Fill]:
        if not self._states:
            self._update_book(book)
            return []
        self._infer_cancels(book)
        fills: list[Fill] = []
        for trade in trades:
            dead: list[str] = []
            for oid, state in self._states.items():
                f = state.process_trade(trade, self._cancel_model)
                if f:
                    fills.append(f)
                if not state.is_active:
                    dead.append(oid)
            for oid in dead:
                self._states.pop(oid, None)
        self._update_book(book)
        return fills

    def _infer_cancels(self, book: OrderBook) -> None:
        # if book size at a price dropped without a matching trade, assume cancels
        # TODO: this over-counts when a trade AND a cancel happen in the same tick
        current = {lvl.price: lvl.size for lvl in book.bids + book.asks}
        for s in self._states.values():
            if not s.is_active:
                continue
            prev = self._prev_book.get(s.order.price, 0.0)
            curr = current.get(s.order.price, 0.0)
            if prev > curr + 1e-12:
                s.qty_in_front = max(0.0, s.qty_in_front - (prev - curr))

    def _update_book(self, book: OrderBook) -> None:
        self._prev_book = {lvl.price: lvl.size for lvl in book.bids + book.asks}


def _trade_matches(trade: Trade, order: Order) -> bool:
    if order.side == Side.BUY:
        return trade.side == Side.SELL and trade.price <= order.price
    return trade.side == Side.BUY and trade.price >= order.price


def _qty_in_front(order: Order, book: OrderBook) -> float:
    # yes this is O(n) on book depth. typical book has <50 levels, calm down
    levels = book.bids if order.side == Side.BUY else book.asks
    total  = 0.0
    for lvl in levels:
        at_ours = abs(lvl.price - order.price) < 1e-9
        better  = (
            (order.side == Side.BUY  and lvl.price > order.price) or
            (order.side == Side.SELL and lvl.price < order.price)
        )
        if at_ours or better:
            total += lvl.size
    return total
