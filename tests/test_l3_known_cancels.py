"""
Regression test for the known_cancels ground-truth path added to
BitfinexL3Exchange / FIFOQueueSimulator.

Scenario replayed:
  snapshot: bid@100 (order 1), ask@105 (order 2), bid@99 (order 4), bid@98 (order 5)
  trade:    SELL 1.0 @ 100.0                          (buffered)
  update:   order 1 removed (is_remove)                -> explained by the trade above,
                                                            must NOT be counted as a cancel
  update:   order 4 removed (is_remove), no trade since -> genuine cancel of 1.5 @ 99.0,
            the last emitted tick                          must be counted

Checks two things:
  1. l3_bitfinex.replay_l3 assigns known_cancels correctly per tick (the
     fill-driven removal contributes nothing, the genuine cancel shows up
     as exactly {99.0: 1.5}).
  2. FIFOQueueSimulator actually consumes that ground truth: an order
     resting at 99.0 has its qty_in_front reduced by exactly 1.5, not by
     whatever the old size-delta heuristic would have inferred.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest

from mmbt.core.types import Order, Side
from mmbt.data.l3_bitfinex import BitfinexL3Exchange, replay_l3
from mmbt.data.exchange import ExchangeMetadata
from mmbt.queue.fifo import FIFOQueueSimulator


def _write_capture(tmp_path: Path) -> Path:
    rows = [
        # snapshot
        dict(ts_recv=1.0, symbol="tBTCUSD", channel="book", msg_type="snapshot",
             order_id=1, price=100.0, amount=2.0, is_remove=False),
        dict(ts_recv=1.0, symbol="tBTCUSD", channel="book", msg_type="snapshot",
             order_id=2, price=105.0, amount=-2.0, is_remove=False),
        dict(ts_recv=1.0, symbol="tBTCUSD", channel="book", msg_type="snapshot",
             order_id=4, price=99.0, amount=1.5, is_remove=False),
        dict(ts_recv=1.0, symbol="tBTCUSD", channel="book", msg_type="snapshot",
             order_id=5, price=98.0, amount=3.0, is_remove=False),
        # trade that will explain order 1's removal (SELL taker hits bid@100)
        dict(ts_recv=2.0, symbol="tBTCUSD", channel="trades", msg_type="trade",
             order_id=999, price=100.0, amount=-1.0, is_remove=False),
        # order 1 removed -- should be treated as fill-driven, not a cancel
        dict(ts_recv=2.5, symbol="tBTCUSD", channel="book", msg_type="update",
             order_id=1, price=0.0, amount=0.0, is_remove=True),
        # order 4 removed with no trade since the last emitted tick -- genuine cancel
        dict(ts_recv=3.0, symbol="tBTCUSD", channel="book", msg_type="update",
             order_id=4, price=0.0, amount=0.0, is_remove=True),
    ]
    df = pd.DataFrame(rows)
    d = tmp_path / "l3"
    d.mkdir()
    df.to_parquet(d / "batch_0.parquet")
    return d


class TestKnownCancelsGroundTruth:
    # Three ticks are emitted from this capture: [0] the snapshot-completion
    # tick (full initial book, before order 1's removal is even applied
    # that's an existing quirk of replay_l3, not something this fix
    # changes), [1] right after order 1's fill-driven removal, [2] right
    # after order 4's genuine cancel.

    def test_snapshot_completion_tick_has_no_cancels_yet(self, tmp_path):
        d = _write_capture(tmp_path)
        ticks = list(replay_l3(d, "tBTCUSD"))
        assert len(ticks) == 3
        assert ticks[0].known_cancels == {}

    def test_fill_explained_removal_not_counted_as_cancel(self, tmp_path):
        d = _write_capture(tmp_path)
        ticks = list(replay_l3(d, "tBTCUSD"))
        # order 1 (bid@100) was removed right after a SELL trade at 100 --
        # that's a fill, not a cancel, so it must not show up here.
        assert ticks[1].known_cancels == {}

    def test_genuine_cancel_counted_with_exact_size(self, tmp_path):
        d = _write_capture(tmp_path)
        ticks = list(replay_l3(d, "tBTCUSD"))
        # order 4 (bid@99, size 1.5) was removed with no trade to explain
        # it  a real cancel, and known_cancels should say exactly that.
        assert ticks[2].known_cancels == {99.0: 1.5}

    def test_exchange_adapter_propagates_known_cancels(self, tmp_path):
        d = _write_capture(tmp_path)
        ex = BitfinexL3Exchange()
        ex.register("tBTCUSD", d, ExchangeMetadata())
        ticks = list(ex.load_ticks("tBTCUSD"))
        assert ticks[2].known_cancels == {99.0: 1.5}

    def test_fifo_queue_uses_ground_truth_not_heuristic(self, tmp_path):
        d = _write_capture(tmp_path)
        ticks = list(replay_l3(d, "tBTCUSD"))

        sim = FIFOQueueSimulator(use_rust=False)
        # A resting buy order at 99.0, sitting behind the full 1.5 displayed there.
        order = Order(order_id=str(uuid.uuid4()), symbol="tBTCUSD",
                       side=Side.BUY, price=99.0, size=0.5, ts=0.0)
        sim.register(order, ticks[1].book)
        assert sim._states[order.order_id].qty_in_front == pytest.approx(1.5)

        # Process the tick where order 4 (also resting at 99.0) gets cancelled.
        sim.process_tick(ticks[2].book, ticks[2].trades, ticks[2].known_cancels)
        # Ground truth says exactly 1.5 was cancelled at 99.0 qty_in_front
        # should drop by exactly that amount, not by whatever the raw
        # book-size delta at 99.0 happens to be.
        assert sim._states[order.order_id].qty_in_front == pytest.approx(0.0)
