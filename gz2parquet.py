# -*- coding: utf-8 -*-
"""
Batch-convert all .csv.gz files in a folder to Parquet format.

Each file keeps its basename:
    /input_folder/sensors_2024.csv.gz  →  /output_folder/sensors_2024.parquet

Expected CSV columns:
    latitude, longitude, time_stamp, type_data, region_name

Just set INPUT_DIR and OUTPUT_DIR below and run.
"""

import gzip
import sys
from pathlib import Path

import pandas as pd


# ── CONFIG — edit these two lines ────────────────────────────────────────────
INPUT_DIR  = r"C:\data\input"   # folder with your .csv.gz files
OUTPUT_DIR = r"C:\data\output"  # folder where .parquet files will be saved
# ─────────────────────────────────────────────────────────────────────────────


DTYPE_MAP = {
    "latitude":    "float64",
    "longitude":   "float64",
    "time_stamp":  "str",
    "type_data":   "str",
    "region_name": "str",
}

EXPECTED_COLUMNS = list(DTYPE_MAP.keys())


def read_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, dtype=DTYPE_MAP)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}")

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True, errors="coerce")
    return df


def export_parquet(df: pd.DataFrame, path: Path) -> float:
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return path.stat().st_size / (1024 ** 2)


def convert_folder(input_dir: Path, output_dir: Path) -> None:
    files = sorted(input_dir.glob("*.csv.gz"))

    if not files:
        print(f"No .csv.gz files found in: {input_dir}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    total  = len(files)
    ok     = 0
    failed = []

    print(f"Found {total} file(s) in '{input_dir}'  →  output: '{output_dir}'\n")
    print(f"{'#':<5} {'File':<40} {'Rows':>10}  {'Size':>9}  Status")
    print("-" * 75)

    for idx, src in enumerate(files, 1):
        stem = src.name.removesuffix(".gz").removesuffix(".csv")
        dst  = output_dir / f"{stem}.parquet"

        try:
            df      = read_csv_gz(src)
            size_mb = export_parquet(df, dst)
            print(f"{idx:<5} {src.name:<40} {len(df):>10,}  {size_mb:>7.2f}MB  ✓")
            ok += 1
        except Exception as exc:
            print(f"{idx:<5} {src.name:<40} {'—':>10}  {'—':>9}  ✗  {exc}")
            failed.append(src.name)

    print("-" * 75)
    print(f"\nCompleted: {ok}/{total} converted successfully.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  • {name}")


# ── Run ───────────────────────────────────────────────────────────────────────
input_dir  = Path(INPUT_DIR)
output_dir = Path(OUTPUT_DIR)

if not input_dir.is_dir():
    print(f"Error: '{input_dir}' is not a valid directory.")
else:
    convert_folder(input_dir, output_dir)