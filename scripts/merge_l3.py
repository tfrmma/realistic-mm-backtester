"""
Consolidates all the small batch .parquet files produced by
bitfinex_l3_listener.py (one per flush) into a single sorted Parquet file.

Handles older book-only captures (missing "channel"/"exchange_ts" columns)
transparently, same as l3_bitfinex.py does at replay time.

Usage:
    python merge_l3_parquets.py ./l3 --output ./l3_merged/tBTCUSD.parquet
    python merge_l3_parquets.py ./l3 --output ./l3_merged/tBTCUSD.parquet --symbol tBTCUSD
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def merge(input_dir: Path, output_path: Path, symbol: str | None = None) -> None:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no .parquet files found in {input_dir}")

    print(f"found {len(files)} files, reading...")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    if symbol:
        before = len(df)
        df = df[df["symbol"] == symbol]
        print(f"filtered to symbol={symbol}: {before} -> {len(df)} rows")

    # Backward compat: older book-only captures lack these columns.
    if "channel" not in df.columns:
        df["channel"] = "book"
    else:
        df["channel"] = df["channel"].fillna("book")

    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"dropped {before - len(df)} exact-duplicate rows")

    df = df.sort_values("ts_recv", kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd")

    print(f"wrote {len(df)} rows -> {output_path}")
    if "symbol" in df.columns:
        print(df["symbol"].value_counts().to_string())
    if "channel" in df.columns:
        print(df["channel"].value_counts().to_string())
    print(f"ts_recv range: {df['ts_recv'].min():.3f} -> {df['ts_recv'].max():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge L3 capture batch files into one Parquet")
    parser.add_argument("input_dir", type=Path, help="Directory containing the batch .parquet files")
    parser.add_argument("--output", type=Path, required=True, help="Path for the merged output file")
    parser.add_argument("--symbol", default=None, help="Optional: keep only this symbol")
    args = parser.parse_args()
    merge(args.input_dir, args.output, args.symbol)


if __name__ == "__main__":
    main()
