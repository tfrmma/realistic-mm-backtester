from __future__ import annotations

from typing import Any

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise ImportError("sweep_plots needs matplotlib: pip install realistic-mm-backtester[dev]") from exc

from mmbt.engine.sweep import SweepResult

_STYLE: dict = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.20,
    "grid.linestyle":    "--",
    "font.family":       "monospace",
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f7f7f7",
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
}

_BLUE   = "#1f77b4"
_ORANGE = "#ff7f0e"
_GRAY   = "#7f7f7f"

_METRIC_FNS: dict[str, Any] = {
    "net_pnl":     lambda r: r.metrics.net_pnl(),
    "sharpe":      lambda r: r.report.sharpe if r.report else float("nan"),
    "win_rate":    lambda r: r.report.win_rate if r.report else float("nan"),
    "adverse_sel": lambda r: r.report.adverse_selection_score if r.report else float("nan"),
}


def param_heatmap(
    results: list[SweepResult],
    x_param: str,
    y_param: str,
    metric: str = "net_pnl",
) -> plt.Figure:
    """2D heatmap of a metric across two parameters."""
    _check_metric(metric)
    valid = [r for r in results if r.is_valid]
    if not valid:
        raise ValueError("no valid sweep results")

    x_vals = sorted({r.params[x_param] for r in valid})
    y_vals = sorted({r.params[y_param] for r in valid})
    grid   = np.full((len(y_vals), len(x_vals)), np.nan)
    fn     = _METRIC_FNS[metric]

    for r in valid:
        xi = x_vals.index(r.params[x_param])
        yi = y_vals.index(r.params[y_param])
        grid[yi, xi] = fn(r)

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, len(x_vals) * 1.2), max(4, len(y_vals) * 0.9)))
        vmax = np.nanmax(np.abs(grid)) or 1.0
        im   = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label=metric)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals])
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"{metric}  [{x_param} x {y_param}]")
        for yi in range(len(y_vals)):
            for xi in range(len(x_vals)):
                v = grid[yi, xi]
                if not np.isnan(v):
                    norm   = (v + vmax) / (2 * vmax)
                    color  = "white" if abs(norm - 0.5) > 0.3 else "black"
                    ax.text(xi, yi, f"{v:.4g}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")
        fig.tight_layout()
    return fig


def pnl_vs_adverse(results: list[SweepResult]) -> plt.Figure:
    """Scatter: net PnL vs adverse selection. Top-left is where you want to be."""
    valid = [r for r in results if r.is_valid and r.report is not None]
    if not valid:
        raise ValueError("no valid results with BacktestReport")

    pnl = np.array([r.metrics.net_pnl() for r in valid])
    adv = np.array([r.report.adverse_selection_score for r in valid])  # type: ignore

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(adv, pnl, alpha=0.75, c=pnl, cmap="RdYlGn",
                        s=60, edgecolors="none", vmin=pnl.min(), vmax=pnl.max())
        plt.colorbar(sc, ax=ax, shrink=0.8, label="net PnL")
        ax.axhline(0, color=_GRAY, lw=0.6, ls="--", alpha=0.6)
        ax.axvline(0, color=_GRAY, lw=0.6, ls="--", alpha=0.6)
        ax.set_xlabel("adverse selection score  (neg = good)")
        ax.set_ylabel("net PnL")
        ax.set_title("PnL vs adverse selection")
        best_idx = int(np.argmax(pnl))
        ax.annotate(
            f"best\n{valid[best_idx].params}",
            xy=(adv[best_idx], pnl[best_idx]),
            xytext=(10, -15), textcoords="offset points",
            fontsize=7, color=_BLUE,
            arrowprops=dict(arrowstyle="->", color=_BLUE, lw=0.8),
        )
        fig.tight_layout()
    return fig


def ranking_table(
    results: list[SweepResult],
    by: str = "net_pnl",
    top_n: int = 10,
) -> plt.Figure:
    """Matplotlib table of top N results. Good enough for quick visual ranking."""
    _check_metric(by)
    valid = [r for r in results if r.is_valid]
    if not valid:
        raise ValueError("no valid results")

    fn     = _METRIC_FNS[by]
    ranked = sorted(valid, key=fn, reverse=(by != "adverse_sel"))[:top_n]

    param_keys = list(ranked[0].params.keys())
    stat_keys  = ["net_pnl", "sharpe", "win_rate", "adverse_sel"]
    col_labels = param_keys + stat_keys + ["elapsed_s"]

    rows = []
    for r in ranked:
        row  = [f"{r.params[k]:.4g}" for k in param_keys]
        row += [f"{_METRIC_FNS[k](r):.5g}" for k in stat_keys]
        row += [f"{r.elapsed_s:.2f}"]
        rows.append(row)

    n_rows = len(rows)
    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.2), max(2.0, 0.4 * n_rows + 0.8)))
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="right")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(range(len(col_labels)))
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#2c3e50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, n_rows + 1):
            shade = "#f0f0f0" if i % 2 == 0 else "white"
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor(shade)
        ax.set_title(f"top {n_rows} results  (by {by})", pad=14, fontsize=10)
        fig.tight_layout()
    return fig


def _check_metric(metric: str) -> None:
    if metric not in _METRIC_FNS:
        raise ValueError(f"unknown metric '{metric}'. choose: {list(_METRIC_FNS)}")
