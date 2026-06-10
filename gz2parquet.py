"""
Batch-convert all .csv.gz files in a folder to Parquet format,
resampling to a fixed rate by rounding timestamps to the nearest
integer second and keeping only rows that fall on a multiple of
NEW_SAMPLING_RATE seconds.

Naming convention:
    sensors_2024.csv.gz  →  sensors_2024_15s.parquet

Expected CSV columns:
    latitude, longitude, time_stamp, type_data, region_name

Just set INPUT_DIR, OUTPUT_DIR, and NEW_SAMPLING_RATE below and run.
"""

import gzip
import sys
from pathlib import Path

import pandas as pd


# ── CONFIG — edit these lines ─────────────────────────────────────────────────
INPUT_DIR         = r"data\raw"   # folder with your .csv.gz files
OUTPUT_DIR        = r"data\parquet"  # folder where .parquet files will be saved
NEW_SAMPLING_RATE = 15                 # desired sampling rate in seconds
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

    df["time_stamp"] = pd.to_datetime(
        df["time_stamp"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    return df


def resample(df: pd.DataFrame, rate_seconds: int) -> pd.DataFrame:
    """
    1. Round each timestamp to the nearest integer second.
    2. Keep only rows whose second value is a multiple of rate_seconds.

    A strike at 14:30:14.7s rounds to 14:30:15 → kept  (15 % 15 == 0)
    A strike at 14:30:07.6s rounds to 14:30:08 → dropped (8 % 15 != 0)

    Multiple strikes rounding to the same tick are ALL kept —
    they are real simultaneous events in different locations.
    """
    # Round to nearest second
    df["time_stamp"] = df["time_stamp"].dt.round("1s")

    # Keep only rows on a 15s tick
    mask = df["time_stamp"].dt.second % rate_seconds == 0
    df = df[mask].copy()

    return df.reset_index(drop=True)


def export_parquet(df: pd.DataFrame, path: Path) -> float:
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return path.stat().st_size / (1024 ** 2)


def convert_folder(input_dir: Path, output_dir: Path, rate_seconds: int) -> None:
    files = sorted(input_dir.glob("*.csv.gz"))

    if not files:
        print(f"No .csv.gz files found in: {input_dir}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    total  = len(files)
    ok     = 0
    failed = []

    print(f"Found {total} file(s) in '{input_dir}'")
    print(f"Sampling rate : {rate_seconds}s  |  Output : '{output_dir}'\n")
    print(f"{'#':<5} {'File':<40} {'Rows in':>10} {'Rows out':>10} {'Kept':>7}  {'Size':>9}  Status")
    print("-" * 92)

    for idx, src in enumerate(files, 1):
        stem = src.name.removesuffix(".gz").removesuffix(".csv")
        dst  = output_dir / f"{stem}_{rate_seconds}s.parquet"

        try:
            df_raw        = read_csv_gz(src)
            df_resampled  = resample(df_raw, rate_seconds)
            size_mb       = export_parquet(df_resampled, dst)
            pct_kept      = 100 * len(df_resampled) / len(df_raw) if len(df_raw) else 0

            print(
                f"{idx:<5} {src.name:<40} "
                f"{len(df_raw):>10,} {len(df_resampled):>10,} "
                f"{pct_kept:>6.1f}%  {size_mb:>7.2f}MB  ✓"
            )
            ok += 1
        except Exception as exc:
            print(f"{idx:<5} {src.name:<40} {'—':>10} {'—':>10} {'—':>7}  {'—':>9}  ✗  {exc}")
            failed.append(src.name)

    print("-" * 92)
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
    convert_folder(input_dir, output_dir, NEW_SAMPLING_RATE)