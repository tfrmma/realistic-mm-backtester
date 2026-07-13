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
from mmbt.reporting.html_export import export_html_dashboard
from mmbt.reporting.metrics import BacktestReport, StrategyMetrics


class _QuoteStrategy(BaseStrategy):
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
        self._bid_id = "b" + str(book.ts)
        self._ask_id = "a" + str(book.ts)
        actions += [
            Order(self._bid_id, self.symbol, Side.BUY,  mid - 1.0, 0.1, is_post_only=True),
            Order(self._ask_id, self.symbol, Side.SELL, mid + 1.0, 0.1, is_post_only=True),
        ]
        return actions


def _ticks(n: int = 2_000, seed: int = 1):
    return TickLoader.synthetic(SyntheticConfig(n_ticks=n, trade_prob=0.3, seed=seed)).to_list()


def _full_report() -> BacktestReport:
    engine = ProBacktestEngine(
        latency_config=LatencyConfig(order_us=400.0, cancel_us=250.0, jitter=0.20),
        snapshot_every=50, seed=0,
    )
    engine.add_strategy("mm", _QuoteStrategy("X"), "X")
    return BacktestReport.from_metrics(engine.run(_ticks())["mm"])


class TestExportHtmlDashboard:
    def test_creates_file(self, tmp_path):
        out = export_html_dashboard(_full_report(), tmp_path / "dash.html")
        assert out.exists()
        assert out.suffix == ".html"

    def test_returns_the_given_path(self, tmp_path):
        p   = tmp_path / "sub" / "dash.html"
        p.parent.mkdir()
        out = export_html_dashboard(_full_report(), p)
        assert out == p

    def test_content_looks_like_html(self, tmp_path):
        out  = export_html_dashboard(_full_report(), tmp_path / "dash.html")
        text = out.read_text()
        assert "<html" in text.lower()
        assert "plotly" in text.lower()

    def test_title_appears_in_output(self, tmp_path):
        out  = export_html_dashboard(_full_report(), tmp_path / "dash.html", title="my custom title")
        assert "my custom title" in out.read_text()

    def test_default_title_uses_symbol(self, tmp_path):
        out  = export_html_dashboard(_full_report(), tmp_path / "dash.html")
        assert "X" in out.read_text()  # symbol="X" from _full_report

    def test_cdn_mode_produces_smaller_file(self, tmp_path):
        report   = _full_report()
        embedded = export_html_dashboard(report, tmp_path / "embedded.html", include_plotlyjs=True)
        cdn      = export_html_dashboard(report, tmp_path / "cdn.html", include_plotlyjs="cdn")
        assert cdn.stat().st_size < embedded.stat().st_size

    def test_no_data_at_all_raises(self, tmp_path):
        empty_report = BacktestReport.from_metrics(StrategyMetrics(symbol="X"))
        with pytest.raises(ValueError, match="nothing to plot"):
            export_html_dashboard(empty_report, tmp_path / "dash.html")

    def test_partial_data_no_latency_still_works(self, tmp_path):
        # BacktestEngine has equity/adverse data but no latency tracking
        # dashboard should still render (with an empty panel), not crash
        engine = BacktestEngine(snapshot_every=50)
        engine.add_strategy("mm", _QuoteStrategy("X"), "X")
        report = BacktestReport.from_metrics(engine.run(_ticks(n=1500))["mm"])
        out = export_html_dashboard(report, tmp_path / "dash.html")
        assert out.exists()
        assert "no latency data" in out.read_text()

    def test_only_equity_no_fills_still_works(self, tmp_path):
        m = StrategyMetrics(symbol="X")
        from mmbt.reporting.metrics import EquitySnapshot
        for i in range(5):
            m.equity_snapshots.append(EquitySnapshot(
                ts=float(i * 1000), realized_pnl=float(i), unrealized_pnl=0.0,
                position=0.0, fees_paid=0.0,
            ))
        report = BacktestReport.from_metrics(m)
        out = export_html_dashboard(report, tmp_path / "dash.html")
        assert out.exists()
        assert "no adverse_scores" in out.read_text()
