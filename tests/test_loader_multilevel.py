from __future__ import annotations

import pytest

from mmbt.data.loader import TickLoader, _level_columns


class TestLevelColumns:
    def test_single_level_only(self):
        cols = {"ts", "bid_px", "bid_sz", "ask_px", "ask_sz"}
        assert _level_columns(cols, "bid") == [("bid_px", "bid_sz")]

    def test_multi_level_ordered(self):
        cols = {"bid_px", "bid_sz", "bid_px_2", "bid_sz_2", "bid_px_3", "bid_sz_3"}
        assert _level_columns(cols, "bid") == [
            ("bid_px", "bid_sz"), ("bid_px_2", "bid_sz_2"), ("bid_px_3", "bid_sz_3"),
        ]

    def test_stops_at_gap(self):
        # bid_px_4 present but bid_px_3 missing -> only levels 1 and 2 count
        cols = {"bid_px", "bid_sz", "bid_px_2", "bid_sz_2", "bid_px_4", "bid_sz_4"}
        assert _level_columns(cols, "bid") == [("bid_px", "bid_sz"), ("bid_px_2", "bid_sz_2")]

    def test_no_columns(self):
        assert _level_columns({"ts"}, "bid") == []


def _write_csv(path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


class TestCSVSingleLevel:
    def test_backward_compatible(self, tmp_path):
        p = tmp_path / "single.csv"
        _write_csv(
            p,
            "ts,bid_px,bid_sz,ask_px,ask_sz",
            ["1000.0,99.0,1.5,101.0,2.0"],
        )
        ticks = TickLoader.from_csv(p, symbol="X").to_list()
        assert len(ticks) == 1
        book = ticks[0].book
        assert len(book.bids) == 1
        assert len(book.asks) == 1
        assert book.bids[0].price == pytest.approx(99.0)
        assert book.asks[0].price == pytest.approx(101.0)


class TestCSVMultiLevel:
    def test_five_levels_each_side(self, tmp_path):
        p = tmp_path / "multi.csv"
        header = "ts,bid_px,bid_sz,ask_px,ask_sz," + ",".join(
            f"bid_px_{n},bid_sz_{n},ask_px_{n},ask_sz_{n}" for n in range(2, 6)
        )
        row = "1000.0,100.0,1.0,101.0,1.0," + ",".join(
            f"{100.0 - n},{n}.0,{101.0 + n},{n}.0" for n in range(2, 6)
        )
        _write_csv(p, header, [row])
        ticks = TickLoader.from_csv(p, symbol="X").to_list()
        book = ticks[0].book
        assert len(book.bids) == 5
        assert len(book.asks) == 5
        # level 1 is best: highest bid, lowest ask
        assert book.bids[0].price == pytest.approx(100.0)
        assert book.bids[4].price == pytest.approx(95.0)   # 100 - 5
        assert book.asks[0].price == pytest.approx(101.0)
        assert book.asks[4].price == pytest.approx(106.0)  # 101 + 5

    def test_shallower_row_truncates_gracefully(self, tmp_path):
        # row 2 has no data past level 2 (NaN) -- should yield a 2-level book, not error
        p = tmp_path / "thin.csv"
        header = "ts,bid_px,bid_sz,ask_px,ask_sz,bid_px_2,bid_sz_2,ask_px_2,ask_sz_2"
        rows = [
            "1000.0,100.0,1.0,101.0,1.0,99.0,2.0,102.0,2.0",
            "2000.0,100.0,1.0,101.0,1.0,,,,",
        ]
        _write_csv(p, header, rows)
        ticks = TickLoader.from_csv(p, symbol="X").to_list()
        assert len(ticks[0].book.bids) == 2
        assert len(ticks[1].book.bids) == 1

    def test_chunked_reads_still_multilevel(self, tmp_path):
        p = tmp_path / "chunked.csv"
        header = "ts,bid_px,bid_sz,ask_px,ask_sz,bid_px_2,bid_sz_2,ask_px_2,ask_sz_2"
        rows = [f"{1000.0 * i},100.0,1.0,101.0,1.0,99.0,2.0,102.0,2.0" for i in range(25)]
        _write_csv(p, header, rows)
        ticks = TickLoader.from_csv(p, symbol="X", chunk_size=7).to_list()
        assert len(ticks) == 25
        assert all(len(t.book.bids) == 2 for t in ticks)


class TestParquetMultiLevel:
    def test_multi_level_round_trip(self, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        p = tmp_path / "multi.parquet"
        df = pd.DataFrame({
            "ts": [1000.0, 2000.0],
            "bid_px": [100.0, 101.0], "bid_sz": [1.0, 1.0],
            "ask_px": [102.0, 103.0], "ask_sz": [1.0, 1.0],
            "bid_px_2": [99.0, 100.0], "bid_sz_2": [2.0, 2.0],
            "ask_px_2": [103.0, 104.0], "ask_sz_2": [2.0, 2.0],
        })
        df.to_parquet(p)
        ticks = TickLoader.from_parquet(p, symbol="X").to_list()
        assert len(ticks) == 2
        assert len(ticks[0].book.bids) == 2
        assert ticks[0].book.bids[1].price == pytest.approx(99.0)
