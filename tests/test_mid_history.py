from __future__ import annotations

import pytest

from mmbt.reporting.mid_history import MidHistoryBuffer


class TestMidHistoryBuffer:
    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            MidHistoryBuffer(capacity=0)

    def test_append_tuple_matches_engine_call_pattern(self):
        # engines call mid_history.append((ts, mid)) -- a single tuple, not two args
        b = MidHistoryBuffer(capacity=10)
        b.append((1000.0, 50_000.0))
        assert len(b) == 1
        ts, mid = b.to_arrays()
        assert ts[0] == pytest.approx(1000.0)
        assert mid[0] == pytest.approx(50_000.0)

    def test_len_before_wrap(self):
        b = MidHistoryBuffer(capacity=100)
        for i in range(30):
            b.append((float(i), float(i)))
        assert len(b) == 30

    def test_len_caps_at_capacity(self):
        b = MidHistoryBuffer(capacity=10)
        for i in range(25):
            b.append((float(i), float(i)))
        assert len(b) == 10

    def test_chronological_order_before_wrap(self):
        b = MidHistoryBuffer(capacity=100)
        for i in range(10):
            b.append((float(i * 1000), float(i)))
        ts, mid = b.to_arrays()
        assert ts.tolist() == [float(i * 1000) for i in range(10)]
        assert mid.tolist() == [float(i) for i in range(10)]

    def test_chronological_order_after_wrap(self):
        # capacity=5, push 12 points (0..11) -> only the last 5 (7..11) survive, in order
        b = MidHistoryBuffer(capacity=5)
        for i in range(12):
            b.append((float(i), float(i) * 10.0))
        ts, mid = b.to_arrays()
        assert ts.tolist() == [7.0, 8.0, 9.0, 10.0, 11.0]
        assert mid.tolist() == [70.0, 80.0, 90.0, 100.0, 110.0]

    def test_iter_yields_tuples(self):
        b = MidHistoryBuffer(capacity=10)
        b.append((1.0, 2.0))
        b.append((3.0, 4.0))
        assert list(b) == [(1.0, 2.0), (3.0, 4.0)]

    def test_empty_to_arrays(self):
        b = MidHistoryBuffer(capacity=10)
        ts, mid = b.to_arrays()
        assert len(ts) == 0
        assert len(mid) == 0

    def test_exact_capacity_no_wrap_yet(self):
        b = MidHistoryBuffer(capacity=5)
        for i in range(5):
            b.append((float(i), float(i)))
        ts, _ = b.to_arrays()
        assert ts.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert len(b) == 5
