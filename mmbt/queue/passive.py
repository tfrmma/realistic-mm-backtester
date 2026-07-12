"""Heuristic fill model for BacktestEngine. Not realistic, use ProBacktestEngine
if fill accuracy matters."""

from __future__ import annotations

from mmbt.core.types import Fill, Order, OrderBook, Side, Trade


class PassiveFillSimulator:
    """
    Fills if a trade crosses our price. Size is pro-rata share of level depth.
    fill_ratio is a conservative discount, 0.5 works well in practice.

    The returned Fill.size can be smaller than order.size (a partial fill) --
    this simulator doesn't track order state across calls, so it's on the
    caller (BacktestEngine._try_fill) to keep a partially-filled order resting
    for the next tick instead of dropping it once *any* fill comes back.
    """

    def __init__(self, fill_ratio: float = 0.5) -> None:
        self.fill_ratio = fill_ratio

    def simulate(self, order: Order, trades: list[Trade], book: OrderBook) -> Fill | None:
        matching = [t for t in trades if _crosses(t, order)]
        if not matching:
            return None

        traded_vol = sum(t.size for t in matching)
        depth      = _level_depth(order, book)

        if depth <= 0:
            fill_size = min(order.size, traded_vol * self.fill_ratio)
        else:
            fill_size = min(order.size, traded_vol * (order.size / depth) * self.fill_ratio)

        if fill_size <= 1e-12:
            return None

        return Fill(
            order_id=order.order_id, symbol=order.symbol,
            side=order.side, price=order.price,
            size=fill_size, is_maker=True, ts=matching[-1].ts,
        )


def _crosses(trade: Trade, order: Order) -> bool:
    if order.side == Side.BUY:
        return trade.side == Side.SELL and trade.price <= order.price
    return trade.side == Side.BUY and trade.price >= order.price


def _level_depth(order: Order, book: OrderBook) -> float:
    levels = book.bids if order.side == Side.BUY else book.asks
    for lvl in levels:
        if abs(lvl.price - order.price) < 1e-9:
            return lvl.size
    return 0.0
