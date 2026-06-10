"""
Batch-convert all .csv.gz files in a folder to Parquet format,
resampling to a fixed rate by rounding timestamps to the nearest
integer second and keeping only rows that fall on a multiple of
NEW_SAMPLING_RATE seconds.

The date is parsed from the filename (e.g. lx_raw_ROGER_20220407_entln.csv.gz
→ 2022-04-07) and used to filter rows to that day only.

Naming convention:
    lx_raw_ROGER_20220407_entln.csv.gz  →  lx_raw_ROGER_20220407_entln_15s.parquet

Expected CSV columns:
    latitude, longitude, time_stamp, type_data

Just set the CONFIG block below and run.
"""

import gzip
import re
import sys
from pathlib import Path

import pandas as pd


# ── CONFIG — edit these lines ─────────────────────────────────────────────────
INPUT_DIR            = r"data\raw"
OUTPUT_DIR           = r"data\parquet"
NEW_SAMPLING_RATE    = 15
VISUALIZATION_PLOTS  = True
TO_UTC               = 0     # hours to add to timestamps to convert to UTC (0 = already UTC)
# ─────────────────────────────────────────────────────────────────────────────


DTYPE_MAP = {
    "latitude":   "float64",
    "longitude":  "float64",
    "time_stamp": "str",
    "type_data":  "Int8",
}

KEEP_COLUMNS = ["latitude", "longitude", "time_stamp", "type_data"]


def parse_date_from_stem(stem: str) -> pd.Timestamp | None:
    """Extract date from filename, e.g. lx_raw_ROGER_20220407_entln → 2022-04-07."""
    match = re.search(r"(\d{8})", stem)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d")
    return None


def read_csv_gz(path: Path, file_date: pd.Timestamp | None) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, dtype=DTYPE_MAP, usecols=KEEP_COLUMNS)

    df["time_stamp"] = pd.to_datetime(
        df["time_stamp"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )

    nat_count = df["time_stamp"].isna().sum()
    if nat_count:
        print(f"         Warning: {nat_count:,} rows with unparseable timestamps dropped")
        df = df.dropna(subset=["time_stamp"])

    # shift timestamps to UTC
    if TO_UTC != 0:
        df["time_stamp"] = df["time_stamp"] + pd.Timedelta(hours=TO_UTC)

    # keep only rows within 00:00:00 to 23:59:59 of the reference day
    if file_date is not None:
        day_start = file_date
        day_end   = file_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
        before    = len(df)
        df        = df[(df["time_stamp"] >= day_start) & (df["time_stamp"] <= day_end)].copy()
        dropped   = before - len(df)
        if dropped:
            print(f"         Info: {dropped:,} rows outside {file_date.date()} removed")

    return df.reset_index(drop=True)


def resample(df: pd.DataFrame, rate_seconds: int) -> pd.DataFrame:
    """
    1. Round each timestamp to the nearest integer second (±0.5s tolerance).
    2. Keep only rows whose second value is a multiple of rate_seconds.
    """
    df = df.copy()
    df["time_stamp"] = df["time_stamp"].dt.round("1s")
    mask = df["time_stamp"].dt.second % rate_seconds == 0
    return df[mask].reset_index(drop=True)


def export_parquet(df: pd.DataFrame, path: Path) -> float:
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return path.stat().st_size / (1024 ** 2)


def convert_folder(input_dir: Path, output_dir: Path, rate_seconds: int) -> list:
    files = sorted(input_dir.glob("*.csv.gz"))

    if not files:
        print(f"No .csv.gz files found in: {input_dir}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    total   = len(files)
    ok      = 0
    failed  = []
    results = []

    print(f"Found {total} file(s) in '{input_dir}'")
    print(f"Sampling rate : {rate_seconds}s  |  Output : '{output_dir}'\n")
    print(f"{'#':<5} {'File':<45} {'Rows in':>10} {'Rows out':>10} {'Kept':>7}  {'Size':>9}  Status")
    print("-" * 98)

    for idx, src in enumerate(files, 1):
        stem      = src.name.removesuffix(".gz").removesuffix(".csv")
        dst       = output_dir / f"{stem}_{rate_seconds}s.parquet"
        file_date = parse_date_from_stem(stem)

        try:
            df_raw       = read_csv_gz(src, file_date)
            df_resampled = resample(df_raw, rate_seconds)
            size_mb      = export_parquet(df_resampled, dst)
            pct_kept     = 100 * len(df_resampled) / len(df_raw) if len(df_raw) else 0

            print(
                f"{idx:<5} {src.name:<45} "
                f"{len(df_raw):>10,} {len(df_resampled):>10,} "
                f"{pct_kept:>6.1f}%  {size_mb:>7.2f}MB  ✓"
            )
            results.append((df_raw, df_resampled, stem))
            ok += 1

        except Exception as exc:
            print(f"{idx:<5} {src.name:<45} {'—':>10} {'—':>10} {'—':>7}  {'—':>9}  ✗  {exc}")
            failed.append(src.name)

    print("-" * 98)
    print(f"\nCompleted: {ok}/{total} converted successfully.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  • {name}")

    return results


# ── Run ───────────────────────────────────────────────────────────────────────
input_dir  = Path(INPUT_DIR)
output_dir = Path(OUTPUT_DIR)

if not input_dir.is_dir():
    print(f"Error: '{input_dir}' is not a valid directory.")
else:
    results = convert_folder(input_dir, output_dir, NEW_SAMPLING_RATE)

    if VISUALIZATION_PLOTS and results:
        print("\nGenerating QA plots...")
        from shared.plot_resample_check import plot_pair
        for df_raw, df_resampled, stem in results:
            plot_pair(df_raw, df_resampled, stem, stem, NEW_SAMPLING_RATE)
        print("Done.")