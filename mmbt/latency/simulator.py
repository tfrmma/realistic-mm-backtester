from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmbt.latency.config import LatencyConfig


@dataclass(order=True)
class _Event:
    arrive_ts: float
    payload: Any = field(compare=False)


class LatencySimulator:
    """
    Min-heap event queue for order/cancel flight times.
    Pass seed for reproducible backtests, each strategy gets its own RNG.
    """

    def __init__(self, config: LatencyConfig, seed: int | None = None) -> None:
        self._cfg  = config
        self._rng  = np.random.default_rng(seed)
        self._heap: list[_Event] = []

    def submit_order(self, order: Any, submit_ts: float) -> float:
        arrive = submit_ts + self._sample(self._cfg.order_us)
        heapq.heappush(self._heap, _Event(arrive, ("order", order)))
        return arrive

    def submit_cancel(self, cancel: Any, submit_ts: float) -> float:
        arrive = submit_ts + self._sample(self._cfg.cancel_us)
        heapq.heappush(self._heap, _Event(arrive, ("cancel", cancel)))
        return arrive

    def feed_delay_us(self) -> float:
        return self._sample(self._cfg.feed_us)

    def poll_arrived(self, now: float) -> list[tuple[str, Any]]:
        arrived: list[tuple[str, Any]] = []
        while self._heap and self._heap[0].arrive_ts <= now:
            arrived.append(heapq.heappop(self._heap).payload)
        return arrived

    def pending_count(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()

    def _sample(self, median: float) -> float:
        return float(self._rng.lognormal(np.log(median), self._cfg.jitter))
