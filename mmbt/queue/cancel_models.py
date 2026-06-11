from __future__ import annotations

from typing import Protocol


class CancelModel(Protocol):
    """Fraction of in-front queue that vanishes alongside each aggressive trade."""
    def cancelled_fraction(self, qty_in_front: float, trade_size: float) -> float: ...


class ReduceRatioCancelModel:
    """Fixed cancel fraction. Simple, fast, works better than you'd expect."""

    def __init__(self, cancel_ratio: float = 0.20) -> None:
        if not 0.0 <= cancel_ratio <= 1.0:
            raise ValueError(f"cancel_ratio must be [0,1], got {cancel_ratio}")
        self.cancel_ratio = cancel_ratio

    def cancelled_fraction(self, qty_in_front: float, trade_size: float) -> float:
        return self.cancel_ratio


class ProbQueueCancelModel:
    """
    Pro-rata: cancel rate scales with trade_size / qty_in_front.
    Clipped to [min_ratio, max_ratio] so thin books don't blow up.
    Tuned on Binance perp data tune it yourself for other venues.
    TODO: per-venue calibration support.
    """

    def __init__(self, min_ratio: float = 0.05, max_ratio: float = 0.70) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def cancelled_fraction(self, qty_in_front: float, trade_size: float) -> float:
        if qty_in_front <= 0:
            return 0.0
        return max(self.min_ratio, min(self.max_ratio, trade_size / qty_in_front))
