from __future__ import annotations

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise ImportError("reporting.plots needs matplotlib: pip install realistic-mm-backtester[dev]") from exc

from mmbt.reporting.metrics import BacktestReport

_STYLE: dict = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "font.family":       "monospace",
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f7f7f7",
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.6,
    "lines.linewidth":   1.2,
}

_GREEN  = "#2ca02c"
_RED    = "#d62728"
_BLUE   = "#1f77b4"
_ORANGE = "#ff7f0e"
_GRAY   = "#7f7f7f"


def equity_curve(report: BacktestReport, ax: plt.Axes | None = None) -> plt.Figure:
    snaps = report.metrics.equity_snapshots
    _require(snaps, "equity_snapshots")
    ts       = [s.ts for s in snaps]
    equity   = [s.equity for s in snaps]
    realized = [s.realized_pnl - s.fees_paid for s in snaps]

    with mpl.rc_context(_STYLE):
        fig, ax_ = _get_ax(ax, figsize=(12, 4))
        ax_.plot(ts, equity,   color=_GREEN, lw=1.4, label="total equity")
        ax_.plot(ts, realized, color=_BLUE,  lw=0.9, ls="--", alpha=0.65, label="realized")
        ax_.axhline(0, color=_GRAY, lw=0.6, alpha=0.5)
        ax_.fill_between(ts, equity, 0, where=[e >= 0 for e in equity], alpha=0.07, color=_GREEN)
        ax_.fill_between(ts, equity, 0, where=[e <  0 for e in equity], alpha=0.07, color=_RED)
        ax_.set_xlabel("time (us)")
        ax_.set_ylabel("PnL")
        ax_.set_title(f"equity curve -- {report.metrics.symbol or 'unknown'}")
        ax_.legend()
        fig.tight_layout()
    return fig


def inventory_over_time(report: BacktestReport, ax: plt.Axes | None = None) -> plt.Figure:
    snaps  = report.metrics.equity_snapshots
    _require(snaps, "equity_snapshots")
    pos    = [s.position for s in snaps]
    colors = [_BLUE if p >= 0 else _RED for p in pos]

    with mpl.rc_context(_STYLE):
        fig, ax_ = _get_ax(ax, figsize=(12, 3))
        ax_.bar(range(len(pos)), pos, color=colors, alpha=0.72, width=1.0, linewidth=0)
        ax_.axhline(0, color=_GRAY, lw=0.7)
        ax_.set_xlabel("snapshot index")
        ax_.set_ylabel("position")
        ax_.set_title("inventory over time")
        fig.tight_layout()
    return fig


def adverse_selection(report: BacktestReport, bins: int = 40, ax: plt.Axes | None = None) -> plt.Figure:
    scores = report.adverse_scores
    _require(scores, "adverse_scores")
    favorable = [s for s in scores if s <= 0]
    adverse   = [s for s in scores if s >  0]
    mean_score = float(np.mean(scores))

    with mpl.rc_context(_STYLE):
        fig, ax_ = _get_ax(ax, figsize=(8, 4))
        if favorable:
            ax_.hist(favorable, bins=max(1, bins // 2), alpha=0.72, color=_GREEN,
                     label="favorable", edgecolor="none")
        if adverse:
            ax_.hist(adverse,   bins=max(1, bins // 2), alpha=0.72, color=_RED,
                     label="adverse", edgecolor="none")
        ax_.axvline(0,          color="black",  lw=0.9, ls="--")
        ax_.axvline(mean_score, color=_ORANGE,  lw=1.2, label=f"mean={mean_score:.4f}")
        ax_.set_xlabel("mid delta after fill (pos = adverse)")
        ax_.set_ylabel("count")
        ax_.set_title("adverse selection")
        ax_.legend()
        fig.tight_layout()
    return fig


def fill_analysis(report: BacktestReport, bins: int = 30, ax: plt.Axes | None = None) -> plt.Figure:
    maker = [f for f in report.metrics.fills if f.is_maker]
    _require(maker, "maker_fills")
    qif = [f.qty_in_front for f in maker]

    with mpl.rc_context(_STYLE):
        fig, ax_ = _get_ax(ax, figsize=(8, 4))
        ax_.hist(qif, bins=bins, color=_BLUE, alpha=0.75, edgecolor="none")
        ax_.axvline(float(np.mean(qif)), color=_ORANGE, lw=1.2, label=f"mean={np.mean(qif):.4f}")
        ax_.set_xlabel("qty in front at fill")
        ax_.set_ylabel("count")
        ax_.set_title("queue position at fill")
        ax_.legend()
        fig.tight_layout()
    return fig


def fill_latency_distribution(
    report: BacktestReport,
    kind: str = "order",
    bins: int = 40,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Histogram of actually-sampled order/cancel latencies vs the theoretical
    lognormal PDF implied by the LatencyConfig used for the run. Only
    ProBacktestEngine populates this data (BacktestEngine/MultiAssetEngine
    have no latency model) sanity-checks that the simulator drew what you
    configured, and shows the realized jitter shape instead of just the
    median/sigma numbers.
    """
    if kind not in ("order", "cancel"):
        raise ValueError(f"kind must be 'order' or 'cancel', got '{kind}'")

    m = report.metrics
    latencies = m.order_latencies_us if kind == "order" else m.cancel_latencies_us
    _require(latencies, f"{kind}_latencies_us (only populated by ProBacktestEngine)")
    if m.latency_config is None:
        raise ValueError("metrics.latency_config is empty only ProBacktestEngine sets it")

    median = m.latency_config.order_us if kind == "order" else m.latency_config.cancel_us
    sigma  = m.latency_config.jitter
    obs_median = float(np.median(latencies))

    with mpl.rc_context(_STYLE):
        fig, ax_ = _get_ax(ax, figsize=(8, 4))
        ax_.hist(latencies, bins=bins, density=True, color=_BLUE, alpha=0.65,
                 edgecolor="none", label="observed")
        if sigma > 0:
            xs  = np.linspace(max(1e-6, min(latencies) * 0.5), max(latencies) * 1.2, 400)
            # lognormal PDF with mu=log(median), sigma=jitter -- matches
            # LatencySimulator._sample()'s rng.lognormal(log(median), jitter)
            pdf = (1.0 / (xs * sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(xs) - np.log(median)) ** 2) / (2 * sigma ** 2)
            )
            ax_.plot(xs, pdf, color=_ORANGE, lw=1.6, label=f"lognormal model (median={median:.0f}us)")
        else:
            ax_.axvline(median, color=_ORANGE, lw=1.6, label=f"model (deterministic, ={median:.0f}us)")
        ax_.axvline(obs_median, color=_RED, lw=1.0, ls="--", label=f"observed median={obs_median:.1f}us")
        ax_.set_xlabel(f"{kind} latency (us)")
        ax_.set_ylabel("density")
        ax_.set_title(f"{kind} latency: observed vs lognormal model")
        ax_.legend()
        fig.tight_layout()
    return fig


def summary_dashboard(report: BacktestReport) -> plt.Figure:
    with mpl.rc_context(_STYLE):
        fig = plt.figure(figsize=(14, 8))
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        _plot_equity(report, ax1)
        _plot_inventory(report, ax2)
        _plot_adverse(report, ax3)
        m = report.metrics
        stats = (
            f"fills: {len(m.fills)}  |  net pnl: {m.net_pnl():.6f}\n"
            f"sharpe: {report.sharpe:.3f}  |  max dd: {report.max_drawdown:.6f}\n"
            f"win rate: {report.win_rate:.2%}  |  adv sel: {report.adverse_selection_score:.4f}"
        )
        fig.text(0.01, 0.98, stats, transform=fig.transFigure,
                 fontsize=8, va="top", family="monospace",
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=4))
        fig.suptitle(f"mmbt -- {m.symbol or 'unknown'}", fontsize=11, y=1.01)
        return fig


def _plot_equity(report: BacktestReport, ax: plt.Axes) -> None:
    snaps = report.metrics.equity_snapshots
    if not snaps:
        ax.text(0.5, 0.5, "no equity snapshots", ha="center", va="center",
                transform=ax.transAxes, color=_GRAY)
        return
    ts      = [s.ts for s in snaps]
    equity  = [s.equity for s in snaps]
    realized = [s.realized_pnl - s.fees_paid for s in snaps]
    ax.plot(ts, equity,   color=_GREEN, lw=1.3, label="total equity")
    ax.plot(ts, realized, color=_BLUE,  lw=0.8, ls="--", alpha=0.6, label="realized")
    ax.axhline(0, color=_GRAY, lw=0.5, alpha=0.5)
    ax.fill_between(ts, equity, 0, where=[e >= 0 for e in equity], alpha=0.07, color=_GREEN)
    ax.fill_between(ts, equity, 0, where=[e <  0 for e in equity], alpha=0.07, color=_RED)
    ax.set_title("equity curve")
    ax.set_ylabel("PnL")
    ax.legend(loc="upper left")


def _plot_inventory(report: BacktestReport, ax: plt.Axes) -> None:
    snaps = report.metrics.equity_snapshots
    if not snaps:
        ax.text(0.5, 0.5, "no snapshots", ha="center", va="center",
                transform=ax.transAxes, color=_GRAY)
        return
    pos    = [s.position for s in snaps]
    colors = [_BLUE if p >= 0 else _RED for p in pos]
    ax.bar(range(len(pos)), pos, color=colors, alpha=0.7, width=1.0, linewidth=0)
    ax.axhline(0, color=_GRAY, lw=0.6)
    ax.set_title("inventory")
    ax.set_ylabel("position")


def _plot_adverse(report: BacktestReport, ax: plt.Axes) -> None:
    scores = report.adverse_scores
    if not scores:
        ax.text(0.5, 0.5, "no adverse selection data",
                ha="center", va="center", transform=ax.transAxes, color=_GRAY, fontsize=8)
        return
    favorable = [s for s in scores if s <= 0]
    adverse   = [s for s in scores if s >  0]
    if favorable:
        ax.hist(favorable, bins=20, alpha=0.7, color=_GREEN, edgecolor="none", label="fav.")
    if adverse:
        ax.hist(adverse,   bins=20, alpha=0.7, color=_RED,   edgecolor="none", label="adv.")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(float(np.mean(scores)), color=_ORANGE, lw=1.1,
               label=f"u={np.mean(scores):.3f}")
    ax.set_title("adverse selection")
    ax.set_xlabel("mid delta after fill")
    ax.legend(loc="upper right")


def _get_ax(ax: plt.Axes | None, figsize: tuple) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.get_figure(), ax
    fig, ax_ = plt.subplots(figsize=figsize)
    return fig, ax_


def _require(collection: list, name: str) -> None:
    if not collection:
        raise ValueError(f"BacktestReport.{name} is empty")
