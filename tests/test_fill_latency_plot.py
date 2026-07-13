from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from mmbt.core.protocol import BaseStrategy
from mmbt.core.types import CancelOrder, Order, OrderBook, Side, Trade
from mmbt.data import SyntheticConfig, TickLoader
from mmbt.engine.pro import ProBacktestEngine
from mmbt.engine.simple import BacktestEngine
from mmbt.latency.config import LatencyConfig
from mmbt.latency.simulator import LatencySimulator
from mmbt.reporting import plots
from mmbt.reporting.metrics import BacktestReport


class _QuoteStrategy(BaseStrategy):
    """Re-quotes every tick exercises both order and cancel submission."""
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._bid_id: str | None = None
        self._ask_id: str | None = None

    def on_tick(self, book: OrderBook, trades: list[Trade]) -> list:
        actions: list = []
        if self._bid_id:
            actions.append(CancelOrder(self._bid_id, self.symbol))
        if self._ask_id:
            actions.append(CancelOrder(self._ask_id, self.symbol))
        mid = book.mid
        self._bid_id = "b" + str(id(book)) + str(book.ts)
        self._ask_id = "a" + str(id(book)) + str(book.ts)
        actions += [
            Order(self._bid_id, self.symbol, Side.BUY,  mid - 1.0, 0.1, is_post_only=True),
            Order(self._ask_id, self.symbol, Side.SELL, mid + 1.0, 0.1, is_post_only=True),
        ]
        return actions


def _ticks(n: int = 2_000, seed: int = 1):
    return TickLoader.synthetic(SyntheticConfig(n_ticks=n, trade_prob=0.3, seed=seed)).to_list()


class TestLatencySimulatorTracking:
    def test_order_latencies_recorded(self):
        sim = LatencySimulator(LatencyConfig(order_us=500.0), seed=0)
        for i in range(10):
            sim.submit_order("x", float(i))
        assert len(sim.order_latencies) == 10
        assert all(d > 0 for d in sim.order_latencies)

    def test_cancel_latencies_recorded(self):
        sim = LatencySimulator(LatencyConfig(cancel_us=300.0), seed=0)
        for i in range(5):
            sim.submit_cancel("c", float(i))
        assert len(sim.cancel_latencies) == 5

    def test_feed_delay_not_tracked(self):
        # feed_delay_us() is sampled every tick regardless of orders would
        # be a firehose of irrelevant data if it were tracked the same way
        sim = LatencySimulator(LatencyConfig(), seed=0)
        for _ in range(20):
            sim.feed_delay_us()
        assert sim.order_latencies == []
        assert sim.cancel_latencies == []

    def test_empty_before_any_submission(self):
        sim = LatencySimulator(LatencyConfig(), seed=0)
        assert sim.order_latencies == []
        assert sim.cancel_latencies == []


class TestEngineTracksLatencies:
    def test_pro_engine_populates_metrics(self):
        engine = ProBacktestEngine(
            latency_config=LatencyConfig(order_us=400.0, cancel_us=250.0, jitter=0.15),
            seed=0,
        )
        engine.add_strategy("mm", _QuoteStrategy("X"), "X")
        m = engine.run(_ticks())["mm"]
        assert len(m.order_latencies_us) > 0
        assert len(m.cancel_latencies_us) > 0
        assert m.latency_config is not None
        assert m.latency_config.order_us == pytest.approx(400.0)

    def test_simple_engine_has_no_latency_data(self):
        engine = BacktestEngine()
        engine.add_strategy("mm", _QuoteStrategy("X"), "X")
        m = engine.run(_ticks(n=500))["mm"]
        assert m.order_latencies_us == []
        assert m.cancel_latencies_us == []
        assert m.latency_config is None


class TestFillLatencyDistributionPlot:
    def _pro_report(self, seed: int = 0) -> BacktestReport:
        engine = ProBacktestEngine(
            latency_config=LatencyConfig(order_us=400.0, cancel_us=250.0, jitter=0.20),
            seed=seed,
        )
        engine.add_strategy("mm", _QuoteStrategy("X"), "X")
        return BacktestReport.from_metrics(engine.run(_ticks())["mm"])

    def test_order_kind_builds(self):
        fig = plots.fill_latency_distribution(self._pro_report(), kind="order")
        assert fig is not None

    def test_cancel_kind_builds(self):
        fig = plots.fill_latency_distribution(self._pro_report(), kind="cancel")
        assert fig is not None

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind must be"):
            plots.fill_latency_distribution(self._pro_report(), kind="bogus")

    def test_no_latency_data_raises(self):
        engine = BacktestEngine()
        engine.add_strategy("mm", _QuoteStrategy("X"), "X")
        report = BacktestReport.from_metrics(engine.run(_ticks(n=500))["mm"])
        with pytest.raises(ValueError):
            plots.fill_latency_distribution(report, kind="order")

    def test_plot_contains_observed_and_model_traces(self):
        fig = plots.fill_latency_distribution(self._pro_report(), kind="order")
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        assert any("model" in l for l in labels)
        assert any("observed" in l for l in labels)

    def test_zero_jitter_still_plots(self):
        engine = ProBacktestEngine(
            latency_config=LatencyConfig(order_us=400.0, cancel_us=250.0, jitter=0.0),
            seed=0,
        )
        engine.add_strategy("mm", _QuoteStrategy("X"), "X")
        report = BacktestReport.from_metrics(engine.run(_ticks(n=500))["mm"])
        fig = plots.fill_latency_distribution(report, kind="order")
        assert fig is not None
