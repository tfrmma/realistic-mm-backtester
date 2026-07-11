from __future__ import annotations

import uuid

import pytest

from mmbt.core.portfolio import Portfolio
from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import (
    BookLevel, CancelOrder, Fill, MarketTick,
    Order, OrderBook, Side, Trade,
)
from mmbt.data import SyntheticConfig, TickLoader
from mmbt.engine.pro import ProBacktestEngine
from mmbt.engine.simple import BacktestEngine
from mmbt.latency.config import LatencyConfig
from mmbt.queue.cancel_models import ReduceRatioCancelModel
from mmbt.reporting.metrics import BacktestReport, StrategyMetrics


class _PassiveStrategy(BaseStrategy):
    """Posts one resting bid far below mid. Only fills on big moves."""
    def __init__(self, symbol: str, offset: float = 10.0) -> None:
        self.symbol  = symbol
        self.offset  = offset
        self._posted = False

    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list:
        if self._posted:
            return []
        self._posted = True
        return [Order(str(uuid.uuid4()), self.symbol, Side.BUY,
                      book.mid - self.offset, 0.1, is_post_only=True)]


class _QuoteStrategy(BaseStrategy):
    """Re-quotes tight spread every tick. Exercises cancel + re-quote path."""
    def __init__(self, symbol: str) -> None:
        self.symbol  = symbol
        self._bid_id: str | None = None
        self._ask_id: str | None = None

    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list:
        actions: list = []
        if self._bid_id:
            actions.append(CancelOrder(self._bid_id, self.symbol))
        if self._ask_id:
            actions.append(CancelOrder(self._ask_id, self.symbol))
        mid = book.mid
        self._bid_id = str(uuid.uuid4())
        self._ask_id = str(uuid.uuid4())
        actions += [
            Order(self._bid_id, self.symbol, Side.BUY,  mid - 1.0, 0.1, is_post_only=True),
            Order(self._ask_id, self.symbol, Side.SELL, mid + 1.0, 0.1, is_post_only=True),
        ]
        return actions


def _ticks(n: int = 2_000, seed: int = 42) -> list[MarketTick]:
    return TickLoader.synthetic(SyntheticConfig(n_ticks=n, vol_per_tick=3.0, trade_prob=0.30, seed=seed)).to_list()


def _pro(seed: int = 0) -> ProBacktestEngine:
    return ProBacktestEngine(
        latency_config=LatencyConfig(order_us=400.0, cancel_us=250.0, jitter=0.15),
        cancel_model=ReduceRatioCancelModel(0.15),
        fee_rate=0.0001, snapshot_every=50, seed=seed,
    )


class TestDeterminism:
    def test_same_seed_same_fills(self):
        ticks = _ticks()
        e1 = _pro(seed=7)
        e1.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        r1 = e1.run(ticks)["mm"]
        e2 = _pro(seed=7)
        e2.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        r2 = e2.run(ticks)["mm"]
        assert len(r1.fills) == len(r2.fills)
        assert r1.realized_pnl == pytest.approx(r2.realized_pnl)

    def test_different_seeds_may_differ(self):
        # _pro()'s default latency (order_us=400, cancel_us=250) never approaches
        # the 1000us synthetic tick spacing, so every sampled arrival lands in the
        # same next tick regardless of seed -> fills come out identical. That's a
        # real property of the tick-quantized engine (see pro.py's stale-book TODO),
        # not proof the RNG is broken. To actually exercise seed-driven divergence
        # here, use latency on the same order as the tick spacing so the sampled
        # value can flip which tick an order registers in.
        ticks = _ticks(n=5_000)

        def _wide_latency_pro(seed: int) -> ProBacktestEngine:
            return ProBacktestEngine(
                latency_config=LatencyConfig(order_us=1200.0, cancel_us=900.0, jitter=0.35),
                cancel_model=ReduceRatioCancelModel(0.15),
                fee_rate=0.0001, snapshot_every=50, seed=seed,
            )

        e1 = _wide_latency_pro(seed=1); e1.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        e2 = _wide_latency_pro(seed=2); e2.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        r1, r2 = e1.run(ticks)["mm"], e2.run(ticks)["mm"]
        assert (len(r1.fills) != len(r2.fills)) or (r1.fees_paid != r2.fees_paid)


class TestPnLAccounting:
    def test_net_pnl_consistent(self):
        e = _pro(); e.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        m = e.run(_ticks())["mm"]
        assert m.net_pnl() == pytest.approx(m.realized_pnl - m.fees_paid)

    def test_equity_snapshot_consistent(self):
        e = _pro(); e.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        for s in e.run(_ticks())["mm"].equity_snapshots:
            assert s.equity == pytest.approx(s.realized_pnl + s.unrealized_pnl - s.fees_paid)

    def test_zero_fills_zero_pnl(self):
        e = _pro(); e.add_strategy("mm", _PassiveStrategy("BTC-USD", offset=9999.0), "BTC-USD")
        m = e.run(_ticks())["mm"]
        assert m.realized_pnl == pytest.approx(0.0)

    def test_maker_rebate_non_positive_fees(self):
        # fee_rate is the base (positive) rate. InventoryState.apply_fill already
        # negates it for maker fills (fees_paid += -fee if is_maker), so the rebate
        # falls out on its own -- passing a negative fee_rate here double-flips the
        # sign and produces a *positive* fees_paid, which is what was failing.
        e = ProBacktestEngine(
            latency_config=LatencyConfig(order_us=200.0, jitter=0.10),
            fee_rate=0.0001, snapshot_every=50, seed=99,
        )
        e.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        m = e.run(_ticks(n=5_000))["mm"]
        if m.fills:
            assert m.fees_paid <= 0.0


class TestEngineComparison:
    def test_simple_fills_at_least_as_much(self):
        ticks = _ticks(n=3_000)
        se = BacktestEngine(fee_rate=0.0001, snapshot_every=50)
        se.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        sm = se.run(ticks)["mm"]
        pe = _pro(seed=0); pe.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        pm = pe.run(ticks)["mm"]
        assert len(sm.fills) >= len(pm.fills)

    def test_both_return_strategy_metrics(self):
        ticks = _ticks()
        se = BacktestEngine(snapshot_every=50); se.add_strategy("mm", _PassiveStrategy("X"), "X")
        assert isinstance(se.run(ticks)["mm"], StrategyMetrics)
        pe = _pro(); pe.add_strategy("mm", _PassiveStrategy("X"), "X")
        assert isinstance(pe.run(ticks)["mm"], StrategyMetrics)


class TestBacktestReport:
    def _report(self) -> BacktestReport:
        e = _pro(seed=5); e.add_strategy("mm", _QuoteStrategy("BTC-USD"), "BTC-USD")
        return BacktestReport.from_metrics(e.run(_ticks(n=3_000))["mm"], lookback_ticks=10)

    def test_builds(self):
        assert self._report() is not None

    def test_max_drawdown_non_negative(self):
        assert self._report().max_drawdown >= 0.0

    def test_win_rate_in_range(self):
        wr = self._report().win_rate
        assert 0.0 <= wr <= 1.0

    def test_print_no_crash(self, capsys):
        self._report().print_summary()
        assert "mmbt backtest" in capsys.readouterr().out


class TestPortfolio:
    def test_creates_on_access(self):
        pf = Portfolio()
        assert pf.get("BTC-USD").symbol == "BTC-USD"

    def test_apply_fill(self):
        pf   = Portfolio()
        fill = Fill(str(uuid.uuid4()), "BTC-USD", Side.BUY, 50_000.0, 1.0, True, 0.0)
        pf.apply_fill(fill, 0.0)
        assert pf.get("BTC-USD").position == pytest.approx(1.0)

    def test_total_realized_aggregates(self):
        pf = Portfolio()
        for sym in ("A", "B", "C"):
            pf.get(sym).realized_pnl = 1.0
        assert pf.total_realized_pnl() == pytest.approx(3.0)

    def test_total_pnl(self):
        pf = Portfolio()
        pf.get("X").position  = 2.0
        pf.get("X").avg_entry = 100.0
        assert pf.total_pnl({"X": 110.0}) == pytest.approx(20.0)
