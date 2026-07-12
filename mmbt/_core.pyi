"""
Type stubs for mmbt._core, the compiled Rust extension.
If you see this in your IDE, the Rust extension isn't compiled yet.
Build it with: maturin develop --release
"""

from typing import List, Tuple


class FIFOQueueCore:
    """
    Rust-accelerated FIFO queue simulator.
    Mirrors FIFOQueueSimulator's internal API exactly.
    Instantiated and managed by FIFOQueueSimulator — don't use directly.
    """

    def __init__(
        self,
        cancel_ratio: float | None = None,
        min_ratio:    float | None = None,
        max_ratio:    float | None = None,
    ) -> None:
        """
        Pass EITHER cancel_ratio (ReduceRatioCancelModel) OR min_ratio+max_ratio
        together (ProbQueueCancelModel), never a mix. No args = ReduceRatio
        default of 0.20, matching the Python default.
        """
        ...

    def register(
        self,
        order_id:     str,
        symbol:       str,
        side:         int,    # 1 = BUY, -1 = SELL
        price:        float,
        size:         float,
        ts:           float,
        qty_in_front: float,
    ) -> None: ...

    def cancel(self, order_id: str) -> bool: ...

    def process_tick(
        self,
        bids:   List[Tuple[float, float]],
        asks:   List[Tuple[float, float]],
        trades: List[Tuple[float, float, int, float]],
    ) -> List[Tuple[str, str, int, float, float, bool, float, float, float]]:
        """
        Returns fills as tuples:
          (order_id, symbol, side, price, size, is_maker, ts, qty_in_front, queue_displacement_us)
        """
        ...

    def pending_count(self) -> int: ...
    def active_order_ids(self) -> List[str]: ...
