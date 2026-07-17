"""
Reconstructs an in-memory limit order book AND trade tape from Parquet
files produced by `bitfinex_l3_listener.py` (Bitfinex Raw Book / R0 +
trades capture), and replays them as a stream of MarketTick, implementing
the Exchange protocol so it drops straight into BacktestEngine /
ProBacktestEngine like CSVExchange does.

Merge strategy:
Book rows (channel="book") and trade rows (channel="trades", msg_type="trade")
are merged by `ts_recv` (local receive wall-clock), not by `seq` `seq` is
a single counter Bitfinex shares across every channel on the connection
(when SEQ_ALL is enabled), so it reflects server dispatch order, not the
per-row chronological order needed for a correct merge here.
Trades are buffered as they arrive and attached to whichever book-driven
tick is emitted next i.e. each MarketTick.trades holds the prints that
happened since the previous emitted tick. This is an approximation (trade
prints aren't re-ordered relative to each other, but their exact placement
relative to a same-instant book update is receive-order, not exchange-order),
good enough for FIFO queue / cancel-model purposes.

Backward compatible with book-only captures (no "trades" channel, no
"channel"/"exchange_ts" columns): missing "channel" is treated as "book".

Session handling:
Each WebSocket (re)connection starts a fresh `seq` count per channel and
always begins with a fresh book snapshot. This module detects a new
snapshot arriving after the book has already been built and clears prior
state before applying it so re-runs across a dropped connection replay
correctly instead of carrying stale orders forward.

Usage:
    from mmbt.data.l3_bitfinex import BitfinexL3Exchange
    from mmbt.data.exchange import ExchangeMetadata

    ex = BitfinexL3Exchange()
    ex.register("tBTCUSD", "./l3", ExchangeMetadata(tick_size=0.1))
    for tick in ex.load_ticks("tBTCUSD"):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from pathlib import Path
from typing import Iterator

import pandas as pd

from mmbt.core.types import BookLevel, MarketTick, OrderBook, Side, Trade
from mmbt.data.exchange import ExchangeMetadata


def _load_symbol_frame(directory: Path, symbol: str) -> pd.DataFrame:
    """Concatenate every capture file in `directory`, filter to one symbol,
    and sort into strict replay order.

    Files are named bitfinex_l3_<unix_ts>_<counter>.parquet by the listener,
    so lexicographic filename order already matches capture order but we
    sort by (ts_recv, seq) explicitly rather than relying on that, since
    it's the field that actually encodes ordering.
    """
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no .parquet files found in {directory}")

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["symbol"] == symbol]
    if df.empty:
        raise ValueError(f"no rows for symbol '{symbol}' in {directory}")

    if "channel" not in df.columns:
        df["channel"] = "book"
    else:
        df["channel"] = df["channel"].fillna("book")

    # Sort by ts_recv only, stable. seq (from Bitfinex's SEQ_ALL) is a
    # single counter shared across every channel on the connection, not
    # independent per channel but it's still not a useful sort key here
    # since it only tells you server-side dispatch order, not receive
    # order, and ts_recv already captures true arrival order for our
    # purposes. A stable sort preserves original capture order (single-
    # threaded receive loop) for genuine ts_recv ties.
    df = df.sort_values("ts_recv", kind="stable").reset_index(drop=True)
    return df


@dataclass
class _LiveBook:
    """order_id -> (price, size, side_sign) for one symbol, plus the
    aggregation into BookLevel lists that OrderBook needs."""

    depth: int
    _orders: dict[int, tuple[float, float, int]] = field(default_factory=dict)

    def clear(self) -> None:
        self._orders.clear()

    def apply(self, order_id: int, price: float, amount: float, is_remove: bool) -> None:
        if is_remove:
            self._orders.pop(order_id, None)
            return
        side_sign = 1 if amount > 0 else -1
        self._orders[order_id] = (price, abs(amount), side_sign)

    def levels(self) -> tuple[list[BookLevel], list[BookLevel]] | None:
        bid_agg: dict[float, float] = {}
        ask_agg: dict[float, float] = {}
        for price, size, side_sign in self._orders.values():
            bucket = bid_agg if side_sign > 0 else ask_agg
            bucket[price] = bucket.get(price, 0.0) + size

        if not bid_agg or not ask_agg:
            return None  # book not two-sided yet, can't form a valid tick

        bids = sorted(bid_agg.items(), key=lambda kv: -kv[0])[: self.depth]
        asks = sorted(ask_agg.items(), key=lambda kv: kv[0])[: self.depth]
        bid_levels = [BookLevel(price=p, size=s) for p, s in bids]
        ask_levels = [BookLevel(price=p, size=s) for p, s in asks]

        if bid_levels[0].price >= ask_levels[0].price:
            return None  # crossed/self-inconsistent snapshot mid-build, skip

        return bid_levels, ask_levels


def _removal_explained_by_trade(price: float, side_sign: int, trades: list[Trade]) -> bool:
    """True if a buffered trade looks like it's what emptied this order out
    of the book (a fill), as opposed to a genuine cancel.

    Bitfinex R0 signals order removal the same way (price == 0) whether the
    order was fully filled or outright cancelled -- the book alone can't
    tell you which. We approximate the distinction using the trades channel:
    a resting bid (side_sign > 0) gets taken out by a SELL-side taker print
    at that exact price; a resting ask (side_sign < 0) gets taken out by a
    BUY-side taker print. This is a heuristic, not a guarantee (e.g. it
    can't distinguish "this exact order got filled" from "some other order
    at the same price got filled and this one was cancelled a moment
    later") -- but getting this wrong in either direction is still better
    than the alternative of treating every removal as a cancel, which
    double-counts every fill (once via the trade-driven consumption in
    FIFOQueueState.process_trade, once via known_cancels).
    """
    want_side = Side.SELL if side_sign > 0 else Side.BUY
    return any(t.side == want_side and abs(t.price - price) < 1e-9 for t in trades)


def replay_l3(directory: str | Path, symbol: str, depth: int = 10) -> Iterator[MarketTick]:
    """Core replay generator: reads captured book+trades rows for `symbol`
    and yields one MarketTick per exchange message that changed the book
    (heartbeats and checksum rows carry no book state and are skipped).
    Trade prints buffered since the last emitted tick are attached to the
    next one.
    """
    df = _load_symbol_frame(Path(directory), symbol)
    book = _LiveBook(depth=depth)
    last_msg_type: str | None = None
    trade_buffer: list[Trade] = []
    # Ground-truth cancelled size since the last emitted tick, by price.
    # Populated only from update-phase is_remove events not explained by a
    # same-tick trade (see _removal_explained_by_trade) -- snapshot-phase
    # removals are book construction, not real cancellations, and are never
    # added here.
    known_cancels: dict[float, float] = {}

    for row in df.itertuples(index=False):
        if row.channel == "trades":
            if row.msg_type == "trade":
                trade_buffer.append(
                    Trade(
                        price=float(row.price),
                        size=abs(float(row.amount)),
                        side=Side.BUY if row.amount > 0 else Side.SELL,
                        ts=row.ts_recv,
                    )
                )
            continue  # trades never mutate book state directly

        msg_type = row.msg_type

        if msg_type == "snapshot":
            if last_msg_type not in (None, "snapshot"):
                # A fresh snapshot after the book was already built means a
                # reconnect happened discard prior (possibly stale) state,
                # including any cancel tracking accumulated before the drop.
                book.clear()
                known_cancels.clear()
            book.apply(int(row.order_id), float(row.price), float(row.amount), bool(row.is_remove))
            last_msg_type = "snapshot"
            continue

        # Decide whether this row's removal (if any) is a genuine cancel
        # BEFORE the snapshot-completion flush below can clear trade_buffer
        # -- otherwise a trade and the removal it explains, arriving one
        # row apart right at the snapshot/update boundary, would have the
        # trade wiped out from under the check that's supposed to see it.
        pending_cancel: tuple[float, float] | None = None
        if msg_type == "update" and bool(row.is_remove):
            existing = book._orders.get(int(row.order_id))
            if existing is not None:
                rm_price, rm_size, rm_side_sign = existing
                if not _removal_explained_by_trade(rm_price, rm_side_sign, trade_buffer):
                    pending_cancel = (rm_price, rm_size)

        if last_msg_type == "snapshot":
            # Snapshot phase just ended emit exactly one tick for the
            # fully-loaded initial book before applying the next event.
            levels = book.levels()
            if levels is not None:
                bids, asks = levels
                yield MarketTick(
                    book=OrderBook(bids=bids, asks=asks, ts=row.ts_recv),
                    trades=list(trade_buffer), ts=row.ts_recv,
                    known_cancels=dict(known_cancels),
                )
                trade_buffer.clear()
                known_cancels.clear()

        if msg_type == "update":
            if pending_cancel is not None:
                rm_price, rm_size = pending_cancel
                known_cancels[rm_price] = known_cancels.get(rm_price, 0.0) + rm_size
            book.apply(int(row.order_id), float(row.price), float(row.amount), bool(row.is_remove))
            levels = book.levels()
            if levels is not None:
                bids, asks = levels
                yield MarketTick(
                    book=OrderBook(bids=bids, asks=asks, ts=row.ts_recv),
                    trades=list(trade_buffer), ts=row.ts_recv,
                    known_cancels=dict(known_cancels),
                )
                trade_buffer.clear()
                known_cancels.clear()

        # heartbeat / checksum rows: no book mutation, no tick, but don't
        # clear last_msg_type's snapshot-flush role either.
        last_msg_type = msg_type


class BitfinexL3Exchange:
    """Exchange adapter over Bitfinex raw-book Parquet captures.
    Implements the same Exchange protocol as CSVExchange."""

    def __init__(self, name: str = "bitfinex_l3", depth: int = 10,
                 default_metadata: ExchangeMetadata | None = None) -> None:
        self.name = name
        self.depth = depth
        self._dirs: dict[str, Path] = {}
        self._meta: dict[str, ExchangeMetadata] = {}
        self._default = default_metadata or ExchangeMetadata()

    def register(self, symbol: str, directory: str | Path, meta: ExchangeMetadata | None = None) -> None:
        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"capture directory not found: {d}")
        self._dirs[symbol] = d
        if meta:
            self._meta[symbol] = meta

    def load_ticks(self, symbol: str, start_ts: float = -inf, end_ts: float = inf) -> Iterator[MarketTick]:
        directory = self._dirs.get(symbol)
        if directory is None:
            raise ValueError(f"no capture directory registered for '{symbol}'")
        for tick in replay_l3(directory, symbol, depth=self.depth):
            if tick.ts < start_ts:
                continue
            if tick.ts > end_ts:
                break
            yield tick

    def tick_size(self, symbol: str) -> float:
        return self._meta.get(symbol, self._default).tick_size

    def min_order_size(self, symbol: str) -> float:
        return self._meta.get(symbol, self._default).min_order_size

    def registered_symbols(self) -> list[str]:
        return list(self._dirs.keys())

    def __repr__(self) -> str:
        return f"BitfinexL3Exchange(name={self.name!r}, symbols={self.registered_symbols()})"
