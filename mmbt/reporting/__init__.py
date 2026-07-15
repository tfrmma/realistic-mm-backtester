from mmbt.reporting.metrics import (
    BacktestReport,
    EquitySnapshot,
    FillRecord,
    StrategyMetrics,
)
from mmbt.reporting.mid_history import MidHistoryBuffer

__all__ = [
    "BacktestReport", "EquitySnapshot", "FillRecord", "StrategyMetrics",
    "MidHistoryBuffer",
]

# matplotlib and plotly are optional dev dependencies (see pyproject.toml's
# core `dependencies` neither is in there), not needed to run a backtest
# at all. Importing these eagerly and unconditionally would break
# `import mmbt.reporting` and transitively `import mmbt.engine`, which
# pulls in reporting.metrics -- for anyone who did a plain `pip install mmbt`
# without `[dev]`. This pre-existed for plots/sweep_plots; same treatment
# applied to html_export so it doesn't repeat the mistake.
try:
    from mmbt.reporting import plots
    __all__.append("plots")
except ImportError:
    pass
try:
    from mmbt.reporting import sweep_plots
    __all__.append("sweep_plots")
except ImportError:
    pass
try:
    from mmbt.reporting import html_export
    __all__.append("html_export")
except ImportError:
    pass
