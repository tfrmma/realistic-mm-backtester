from __future__ import annotations

from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:
    raise ImportError("reporting.html_export needs plotly: pip install realistic-mm-backtester[dev]") from exc

from mmbt.reporting.metrics import BacktestReport

_GREEN  = "#2ca02c"
_RED    = "#d62728"
_BLUE   = "#1f77b4"
_ORANGE = "#ff7f0e"
_GRAY   = "#7f7f7f"


def export_html_dashboard(
    report: BacktestReport,
    path: str | Path,
    title: str | None = None,
    include_plotlyjs: bool | str = True,
) -> Path:
    """
    Self-contained interactive HTML dashboard equity curve, inventory,
    adverse selection, and fill latency (when available), all zoomable /
    pannable, unlike the static matplotlib plots in reporting.plots. Opens in
    any browser, no server needed.

    include_plotlyjs=True embeds plotly.js (~3-4MB) so the file works fully
    offline and can be e-mailed/shared as-is. Pass "cdn" for a much smaller
    file that loads plotly.js from a CDN instead (needs internet to view).
    """
    m = report.metrics
    snaps       = m.equity_snapshots
    scores      = report.adverse_scores
    has_latency = bool(m.order_latencies_us)
    if not snaps and not scores and not has_latency:
        raise ValueError("nothing to plot report has no equity_snapshots, adverse_scores, or latency data")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Equity curve", "Inventory over time", "Adverse selection", "Fill latency"),
    )

    if snaps:
        ts       = [s.ts for s in snaps]
        equity   = [s.equity for s in snaps]
        realized = [s.realized_pnl - s.fees_paid for s in snaps]
        fig.add_trace(go.Scatter(x=ts, y=equity, name="total equity",
                                  line=dict(color=_GREEN, width=1.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=realized, name="realized",
                                  line=dict(color=_BLUE, width=1.1, dash="dash")), row=1, col=1)

        pos    = [s.position for s in snaps]
        colors = [_BLUE if p >= 0 else _RED for p in pos]
        fig.add_trace(go.Bar(x=list(range(len(pos))), y=pos, marker_color=colors,
                              name="position", showlegend=False), row=1, col=2)
    else:
        _empty_panel(fig, row=1, col=1, text="no equity_snapshots")
        _empty_panel(fig, row=1, col=2, text="no equity_snapshots")

    if scores:
        fig.add_trace(go.Histogram(x=scores, nbinsx=40, name="mid delta after fill",
                                    marker_color=_GRAY, showlegend=False), row=2, col=1)
    else:
        _empty_panel(fig, row=2, col=1, text="no adverse_scores")

    if has_latency:
        fig.add_trace(go.Histogram(x=m.order_latencies_us, nbinsx=40, name="order latency (us)",
                                    marker_color=_ORANGE, showlegend=False), row=2, col=2)
    else:
        _empty_panel(fig, row=2, col=2, text="no latency data\n(ProBacktestEngine only)")

    fig.update_layout(
        title=title or f"mmbt backtest -- {m.symbol or 'unknown'}",
        height=800,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="time (us)", row=1, col=1)
    fig.update_yaxes(title_text="equity", row=1, col=1)
    fig.update_xaxes(title_text="snapshot index", row=1, col=2)
    fig.update_yaxes(title_text="position", row=1, col=2)
    fig.update_xaxes(title_text="mid delta after fill (pos = adverse)", row=2, col=1)
    fig.update_xaxes(title_text="latency (us)", row=2, col=2)

    out = Path(path)
    fig.write_html(str(out), include_plotlyjs=include_plotlyjs)
    return out


def _empty_panel(fig: go.Figure, row: int, col: int, text: str) -> None:
    fig.add_annotation(
        text=text, showarrow=False, xref=f"x{_axis_idx(row, col)} domain",
        yref=f"y{_axis_idx(row, col)} domain", x=0.5, y=0.5,
        font=dict(color=_GRAY, size=11),
    )


def _axis_idx(row: int, col: int, ncols: int = 2) -> str:
    idx = (row - 1) * ncols + col
    return "" if idx == 1 else str(idx)
