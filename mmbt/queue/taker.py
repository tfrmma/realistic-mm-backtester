from __future__ import annotations

from dataclasses import dataclass

from mmbt.core.types import Order, OrderBook, Side


@dataclass(slots=True)
class TakerExecution:
    filled_size: float
    vwap_price: float
    levels_consumed: int


def crosses_book(order: Order, book: OrderBook) -> bool:
    """True if order.price would immediately match the opposite side's best
    level i.e. this order can't rest, it has to take liquidity or reject."""
    if not book.bids or not book.asks:
        return False
    if order.side == Side.BUY:
        return order.price >= book.best_ask.price
    return order.price <= book.best_bid.price


def sweep_book(order: Order, book: OrderBook) -> TakerExecution | None:
    """
    Walk the book on the side opposite the order (asks for a BUY, sells for
    the taker to consume; bids for a SELL), consuming visible depth level by
    level up to order.size, computing a size-weighted average execution
    price. Stops at the first level the order's limit price wouldn't accept,
    or when depth runs out a taker order only fills what the book actually
    shows, same as a real IOC order; it never invents liquidity beyond that.

    Returns None if nothing fills (order doesn't actually cross, or the
    crossed side has zero size).
    """
    levels = book.asks if order.side == Side.BUY else book.bids
    remaining = order.size
    notional  = 0.0
    filled    = 0.0
    consumed  = 0

    for lvl in levels:
        if order.side == Side.BUY and lvl.price > order.price:
            break
        if order.side == Side.SELL and lvl.price < order.price:
            break
        take = min(remaining, lvl.size)
        if take <= 1e-12:
            continue
        notional  += take * lvl.price
        filled    += take
        remaining -= take
        consumed  += 1
        if remaining <= 1e-12:
            break

    if filled <= 1e-12:
        return None
    return TakerExecution(filled_size=filled, vwap_price=notional / filled, levels_consumed=consumed)
