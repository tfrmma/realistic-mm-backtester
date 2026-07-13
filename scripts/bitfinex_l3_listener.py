"""
Captures Bitfinex Raw Book (R0, per-order) + trades over one public WS
connection. Writes normalized Parquet, batched by time/size, gap-safe
across reconnects. seq is connection-wide (SEQ_ALL), not per-channel.

pip install websockets pyarrow
python bitfinex_l3_listener.py --symbols tBTCUSD,tETHUSD --output-dir ./l3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import websockets

WS_URL = "wss://api-pub.bitfinex.com/ws/2"

FLAG_SEQ_ALL = 65536
FLAG_OB_CHECKSUM = 131072
CONF_FLAGS = FLAG_SEQ_ALL | FLAG_OB_CHECKSUM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("bitfinex_l3")

SCHEMA = pa.schema(
    [
        ("ts_recv", pa.float64()),     # local receive time (unix, seconds)
        ("symbol", pa.string()),       # e.g. tBTCUSD
        ("channel", pa.string()),      # "book" | "trades"
        ("chan_id", pa.int64()),
        ("seq", pa.int64()),           # public sequence number, per-channel counter
        ("msg_type", pa.string()),     # snapshot | update | checksum | heartbeat | trade
        ("order_id", pa.int64()),      # book: order id. trades: trade id.
        ("price", pa.float64()),
        ("amount", pa.float64()),
        ("side", pa.string()),         # BUY | SELL, derived from sign(amount)
        ("is_remove", pa.bool_()),     # book only: True when price == 0 (order left book)
        ("checksum", pa.int64()),      # populated only for msg_type == "checksum"
        ("exchange_ts", pa.float64()), # trades only: exchange trade time (unix, seconds)
    ]
)


def _row(**kwargs: Any) -> dict[str, Any]:
    row = {name: None for name in SCHEMA.names}
    row.update(kwargs)
    return row


def normalize_entry(chan_id: int, symbol: str, entry: list, seq: int | None, ts_recv: float) -> dict[str, Any]:
    """Book entry: [order_id, price, amount]"""
    order_id, price, amount = entry
    price = float(price)
    amount = float(amount)
    return _row(
        ts_recv=ts_recv,
        symbol=symbol,
        channel="book",
        chan_id=chan_id,
        seq=seq,
        msg_type="update",
        order_id=int(order_id),
        price=price,
        amount=amount,
        side="BUY" if amount > 0 else "SELL",
        is_remove=(price == 0.0),
    )


def normalize_trade_entry(chan_id: int, symbol: str, entry: list, seq: int | None, ts_recv: float) -> dict[str, Any]:
    """Trade entry: [ID, MTS, AMOUNT, PRICE]. amount > 0 means the taker bought."""
    trade_id, mts, amount, price = entry
    amount = float(amount)
    price = float(price)
    return _row(
        ts_recv=ts_recv,
        symbol=symbol,
        channel="trades",
        chan_id=chan_id,
        seq=seq,
        msg_type="trade",
        order_id=int(trade_id),
        price=price,
        amount=amount,
        side="BUY" if amount > 0 else "SELL",
        exchange_ts=float(mts) / 1000.0,
    )


def _subscribe_msgs(symbol: str) -> list[dict]:
    return [
        {"event": "subscribe", "channel": "book", "prec": "R0", "symbol": symbol},
        {"event": "subscribe", "channel": "trades", "symbol": symbol},
    ]


@dataclass
class Writer:
    output_dir: Path
    batch_size: int = 5000
    flush_interval_s: float = 30.0
    _buffer: list[dict[str, Any]] = field(default_factory=list)
    _last_flush: float = field(default_factory=time.monotonic)
    _file_count: int = 0
    _buffer_first_ts: float | None = field(default=None, init=False)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file_count = self._get_next_file_count()

    def _get_next_file_count(self) -> int:
        existing = list(self.output_dir.glob("bitfinex_l3_*.parquet"))
        if not existing:
            return 0
        
        max_count = -1
        for f in existing:
            try:
                count_str = f.stem.split("_")[-1]
                count = int(count_str)
                if count > max_count:
                    max_count = count
            except (IndexError, ValueError):
                continue
        return max_count + 1

    def add(self, row: dict[str, Any]) -> None:
        if not self._buffer:
            self._buffer_first_ts = row.get("ts_recv") or time.time()
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size or (
            time.monotonic() - self._last_flush >= self.flush_interval_s
        ):
            self.flush()

    def add_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if not self._buffer:
            self._buffer_first_ts = rows[0].get("ts_recv") or time.time()
        self._buffer.extend(rows)
        if len(self._buffer) >= self.batch_size or (
            time.monotonic() - self._last_flush >= self.flush_interval_s
        ):
            self.flush()

    def flush_on_reconnect(self) -> None:
        """Call before processing a new WS session, prevents mixing epochs in one file."""
        if self._buffer:
            log.info("flushing pending buffer on reconnect (%d rows)", len(self._buffer))
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            self._last_flush = time.monotonic()
            return

        table = pa.Table.from_pylist(self._buffer, schema=SCHEMA)
        # filename uses first buffered msg's timestamp, not flush time
        name_ts = int(self._buffer_first_ts) if self._buffer_first_ts else int(time.time())
        fname = f"bitfinex_l3_{name_ts}_{self._file_count:05d}.parquet"
        path = self.output_dir / fname
        pq.write_table(table, path, compression="zstd")
        log.info("flushed %d rows -> %s", len(self._buffer), path)
        
        self._buffer.clear()
        self._buffer_first_ts = None
        self._last_flush = time.monotonic()
        self._file_count += 1

class BitfinexL3Listener:
    def __init__(
        self,
        symbols: list[str],
        output_dir: str | Path,
        batch_size: int = 5000,
        flush_interval_s: float = 30.0,
    ) -> None:
        self.symbols = symbols
        self.writer = Writer(
            output_dir=Path(output_dir),
            batch_size=batch_size,
            flush_interval_s=flush_interval_s,
        )
        self._chan_info: dict[int, tuple[str, str]] = {}  # chan_id -> (channel_type, symbol)
        self._last_seq: int | None = None  # global across the whole connection, not per-channel
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._run_once()
                backoff = 1.0
            except (websockets.ConnectionClosed, OSError) as exc:
                if self._stop.is_set():
                    break
                log.warning("connection lost (%s), reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        self.writer.flush()

    async def _run_once(self) -> None:
        self.writer.flush_on_reconnect()
        self._chan_info.clear()
        self._last_seq = None
        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps({"event": "conf", "flags": CONF_FLAGS}))

            for symbol in self.symbols:
                for sub_msg in _subscribe_msgs(symbol):
                    await ws.send(json.dumps(sub_msg))
            log.info("subscribed to raw book (R0) + trades for %s", self.symbols)

            while not self._stop.is_set():
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
                ts_recv = time.time()
                msg = json.loads(raw)
                self._handle(msg, ts_recv)

    def _handle(self, msg: Any, ts_recv: float) -> None:
        if isinstance(msg, dict):
            event = msg.get("event")
            if event == "subscribed" and msg.get("channel") in {"book", "trades"}:
                self._chan_info[msg["chanId"]] = (msg["channel"], msg["symbol"])
                log.info("chan %s -> %s (%s)", msg["chanId"], msg["symbol"], msg["channel"])
            elif event == "error":
                log.error("exchange error: %s", msg)
            elif event in {"info", "conf"}:
                log.info("control message: %s", msg)
            return

        chan_id = msg[0]
        second = msg[1]
        channel, symbol = self._chan_info.get(chan_id, ("?", "?"))

        if second == "hb":
            seq = msg[2] if len(msg) > 2 else None
            self._check_seq(chan_id, symbol, seq)
            self.writer.add(
                _row(ts_recv=ts_recv, symbol=symbol, channel=channel, chan_id=chan_id,
                     seq=seq, msg_type="heartbeat")
            )
            return

        if channel == "book":
            self._handle_book(chan_id, symbol, second, msg, ts_recv)
        elif channel == "trades":
            self._handle_trades(chan_id, symbol, second, msg, ts_recv)
        else:
            # startup race: data before "subscribed" ack, still track seq
            trailing = msg[-1] if len(msg) > 2 and isinstance(msg[-1], int) else None
            self._check_seq(chan_id, symbol, trailing)
            log.warning("message on unmapped channel (chan_id=%s): %s", chan_id, msg)

    def _handle_book(self, chan_id: int, symbol: str, second: Any, msg: list, ts_recv: float) -> None:
        if second == "cs":
            checksum_value = msg[2] if len(msg) > 2 else None
            seq = msg[3] if len(msg) > 3 else None
            self._check_seq(chan_id, symbol, seq)
            self.writer.add(
                _row(
                    ts_recv=ts_recv, symbol=symbol, channel="book", chan_id=chan_id, seq=seq,
                    msg_type="checksum", checksum=checksum_value,
                )
            )
            return

        payload = second
        seq = msg[2] if len(msg) > 2 else None
        self._check_seq(chan_id, symbol, seq)

        # snapshot: list of triples
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            rows = [normalize_entry(chan_id, symbol, entry, seq, ts_recv) for entry in payload]
            for row in rows:
                row["msg_type"] = "snapshot"
            self.writer.add_many(rows)
            return

        # single update
        if isinstance(payload, list) and len(payload) == 3:
            row = normalize_entry(chan_id, symbol, payload, seq, ts_recv)
            self.writer.add(row)
            return

        log.warning("unrecognized book message shape: %s", msg)

    def _handle_trades(self, chan_id: int, symbol: str, second: Any, msg: list, ts_recv: float) -> None:
        # "tu" dup-confirms "te", not persisted but still consumes a seq slot
        if second == "tu":
            seq = msg[3] if len(msg) > 3 else None
            self._check_seq(chan_id, symbol, seq)
            return

        if second == "te":
            entry = msg[2]
            seq = msg[3] if len(msg) > 3 else None
            self._check_seq(chan_id, symbol, seq)
            row = normalize_trade_entry(chan_id, symbol, entry, seq, ts_recv)
            self.writer.add(row)
            return

        # trade snapshot (history on subscribe), not persisted
        if isinstance(second, list) and second and isinstance(second[0], list):
            seq = msg[2] if len(msg) > 2 else None
            self._check_seq(chan_id, symbol, seq)
            return

        log.warning("unrecognized trades message shape: %s", msg)

    def _check_seq(self, chan_id: int, symbol: str, seq: int | None) -> None:
        # seq is connection-wide, not per-channel
        if seq is None:
            return
        if self._last_seq is not None and seq != self._last_seq + 1:
            log.warning(
                "seq gap: expected %d, got %d on chan %s (%s), resubscribe for fresh snapshot",
                self._last_seq + 1, seq, chan_id, symbol,
            )
        self._last_seq = seq


async def main_async(args: argparse.Namespace) -> None:
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    listener = BitfinexL3Listener(
        symbols=symbols,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        flush_interval_s=args.flush_interval,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, listener.request_stop)
        except NotImplementedError:
            pass  # Windows

    if args.duration:
        async def _timed_stop():
            await asyncio.sleep(args.duration)
            listener.request_stop()

        asyncio.create_task(_timed_stop())

    await listener.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitfinex Raw Book (L3) capture")
    parser.add_argument(
        "--symbols",
        default="tBTCUSD",
        help="Comma-separated Bitfinex symbols, e.g. tBTCUSD,tETHUSD",
    )
    parser.add_argument("--output-dir", default="./data/l3")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--flush-interval", type=float, default=30.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until Ctrl+C)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
