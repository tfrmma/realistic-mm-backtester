from __future__ import annotations

import numpy as np
import pytest

from mmbt.latency.config import LatencyConfig
from mmbt.latency.simulator import LatencySimulator


def _sim(seed: int = 0, order_us: float = 500.0, cancel_us: float = 300.0) -> LatencySimulator:
    return LatencySimulator(LatencyConfig(order_us=order_us, cancel_us=cancel_us, jitter=0.20), seed=seed)


class TestLatencyConfig:
    def test_defaults_positive(self):
        cfg = LatencyConfig()
        assert cfg.feed_us > 0 and cfg.order_us > 0 and cfg.cancel_us > 0

    def test_jitter_bounds(self):
        with pytest.raises(Exception):
            LatencyConfig(jitter=-0.1)
        with pytest.raises(Exception):
            LatencyConfig(jitter=2.1)

    def test_frozen(self):
        with pytest.raises(Exception):
            LatencyConfig().order_us = 999.0  # type: ignore[misc]

    def test_yaml_round_trip(self, tmp_path):
        cfg = LatencyConfig(feed_us=150.0, order_us=600.0, cancel_us=350.0, jitter=0.25)
        path = str(tmp_path / "lat.yaml")
        cfg.to_yaml(path)
        assert LatencyConfig.from_yaml(path) == cfg


class TestLatencySimulator:
    def test_submit_order_future(self):
        assert _sim().submit_order("x", 1_000.0) > 1_000.0

    def test_cancel_faster_than_order_on_average(self):
        sim = _sim(order_us=500.0, cancel_us=100.0)
        o_times = [sim.submit_order("o", 0.0) for _ in range(500)]
        c_times = [sim.submit_cancel("c", 0.0) for _ in range(500)]
        assert np.mean(c_times) < np.mean(o_times)

    def test_poll_arrived_respects_ts(self):
        sim = _sim(seed=1)
        sim.submit_order("early", 0.0)
        sim.submit_order("late", 1_000_000.0)
        arrived = sim.poll_arrived(10_000.0)
        assert len(arrived) >= 1
        for _, payload in arrived:
            assert payload == "early"

    def test_poll_drains_heap(self):
        sim = _sim(seed=2)
        for i in range(10):
            sim.submit_order(f"o{i}", 0.0)
        assert len(sim.poll_arrived(1e9)) == 10
        assert sim.pending_count() == 0

    def test_poll_empty_when_nothing_due(self):
        sim = _sim(seed=3)
        sim.submit_order("future", 1_000_000.0)
        assert sim.poll_arrived(0.0) == []

    def test_clear(self):
        sim = _sim()
        for _ in range(5):
            sim.submit_order("x", 0.0)
        sim.clear()
        assert sim.pending_count() == 0


class TestSeedDeterminism:
    def test_same_seed_same_sequence(self):
        a = [LatencySimulator(LatencyConfig(), seed=42).submit_order("x", 0.0) for _ in range(50)]
        b = [LatencySimulator(LatencyConfig(), seed=42).submit_order("x", 0.0) for _ in range(50)]
        assert a == b

    def test_different_seeds_differ(self):
        a = [LatencySimulator(LatencyConfig(), seed=1).submit_order("x", 0.0) for _ in range(50)]
        b = [LatencySimulator(LatencyConfig(), seed=2).submit_order("x", 0.0) for _ in range(50)]
        assert a != b


class TestLognormalShape:
    def test_median_approx_correct(self):
        sim     = _sim(order_us=500.0, seed=0)
        samples = [sim.submit_order("x", 0.0) for _ in range(2_000)]
        assert abs(float(np.median(samples)) - 500.0) / 500.0 < 0.10

    def test_all_positive(self):
        sim = _sim(seed=0)
        assert all(sim.submit_order("x", 0.0) > 0 for _ in range(500))

    def test_higher_jitter_wider_spread(self):
        s_low  = LatencySimulator(LatencyConfig(order_us=500.0, jitter=0.05), seed=0)
        s_high = LatencySimulator(LatencyConfig(order_us=500.0, jitter=1.00), seed=0)
        low    = [s_low.submit_order("x",  0.0) for _ in range(1_000)]
        high   = [s_high.submit_order("x", 0.0) for _ in range(1_000)]
        assert np.std(high) > np.std(low)
