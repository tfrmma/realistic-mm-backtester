from __future__ import annotations

import pytest


class TestParquetRowGroupStreaming:
    def _multi_row_group_file(self, tmp_path, n_row_groups=3, rows_per_group=10):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        pd = pytest.importorskip("pandas")
        n = n_row_groups * rows_per_group
        df = pd.DataFrame({
            "ts": [float(i * 1000) for i in range(n)],
            "bid_px": [100.0] * n, "bid_sz": [1.0] * n,
            "ask_px": [101.0] * n, "ask_sz": [1.0] * n,
        })
        table = pa.Table.from_pandas(df, preserve_index=False)
        p = tmp_path / "multi_rg.parquet"
        pq.write_table(table, p, row_group_size=rows_per_group)
        return p, n

    def test_multiple_row_groups_all_read_in_order(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        from mmbt.data.loader import TickLoader

        p, n = self._multi_row_group_file(tmp_path, n_row_groups=4, rows_per_group=25)
        assert pq.ParquetFile(p).num_row_groups >= 2  # sanity: fixture really has multiple groups

        ticks = TickLoader.from_parquet(p, symbol="X").to_list()
        assert len(ticks) == n
        assert [t.ts for t in ticks] == [float(i * 1000) for i in range(n)]

    def test_small_batch_size_still_reads_everything(self, tmp_path):
        pytest.importorskip("pyarrow.parquet")
        from mmbt.data.loader import TickLoader

        p, n = self._multi_row_group_file(tmp_path, n_row_groups=2, rows_per_group=20)
        ticks = TickLoader.from_parquet(p, symbol="X", batch_size=7).to_list()
        assert len(ticks) == n

    def test_does_not_load_whole_file_via_pandas(self, tmp_path, monkeypatch):
        pytest.importorskip("pyarrow.parquet")
        pd = pytest.importorskip("pandas")
        from mmbt.data.loader import TickLoader

        p, n = self._multi_row_group_file(tmp_path, n_row_groups=2, rows_per_group=10)

        def _boom(*args, **kwargs):
            raise AssertionError(
                "pd.read_parquet should not be called streaming must go through pyarrow batches"
            )
        monkeypatch.setattr(pd, "read_parquet", _boom)

        ticks = TickLoader.from_parquet(p, symbol="X").to_list()
        assert len(ticks) == n

    def test_missing_columns_raises(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        pd = pytest.importorskip("pandas")
        from mmbt.data.loader import TickLoader

        df = pd.DataFrame({"ts": [1.0], "bid_px": [100.0]})  # missing ask_px/sz etc.
        table = pa.Table.from_pandas(df, preserve_index=False)
        p = tmp_path / "bad.parquet"
        pq.write_table(table, p)
        with pytest.raises(ValueError, match="missing columns"):
            list(TickLoader.from_parquet(p, symbol="X"))
