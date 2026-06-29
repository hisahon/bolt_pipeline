"""
Batch-convert all .csv.gz files in a folder to Parquet format,
resampling to a fixed rate by snapping every timestamp to the nearest
multiple of NEW_SAMPLING_RATE seconds — NO rows are dropped.

Each row's timestamp is rounded to the closest multiple of
NEW_SAMPLING_RATE seconds (e.g. 15 s), so all original data is
preserved and each record is assigned to a "bin" label.

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

    # keep only rows within [00:00:00.000, 00:00:00.000 of next day)
    # using exclusive upper bound so milliseconds up to 23:59:59.999 are preserved
    if file_date is not None:
        day_start = file_date
        day_end   = file_date + pd.Timedelta(days=1)   # exclusive: < next day midnight
        before    = len(df)
        df        = df[(df["time_stamp"] >= day_start) & (df["time_stamp"] < day_end)].copy()
        dropped   = before - len(df)
        if dropped:
            print(f"         Info: {dropped:,} rows outside {file_date.date()} removed")

    return df.reset_index(drop=True)


def resample(df: pd.DataFrame, rate_seconds: int) -> pd.DataFrame:
    """
    Snap every timestamp to the nearest multiple of rate_seconds.

    Strategy:
        1. Round to the nearest integer second (±0.5 s tolerance).
        2. Compute seconds-of-day for each row.
        3. Round those seconds to the nearest multiple of rate_seconds
           using standard round-half-up arithmetic.
        4. Reconstruct the binned timestamp from date + binned seconds.

    All rows are kept — nothing is filtered out.
    The original sub-second timestamp is preserved in 'time_stamp_orig'.
    """
    df = df.copy()

    # preserve original timestamp
    df["time_stamp_orig"] = df["time_stamp"]

    # step 1 — round to nearest integer second
    ts_sec = df["time_stamp"].dt.round("1s")

    # step 2 — extract the date part (midnight of that day)
    midnight = ts_sec.dt.normalize()

    # step 3 — total seconds since midnight, then bin to nearest multiple
    seconds_of_day = (ts_sec - midnight).dt.total_seconds().astype("int64")
    binned_seconds = ((seconds_of_day + rate_seconds / 2) // rate_seconds * rate_seconds).astype("int64")

    # step 4 — rebuild timestamp from midnight + binned offset
    df["time_stamp"] = midnight + pd.to_timedelta(binned_seconds, unit="s")

    return df.reset_index(drop=True)


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
    print(f"Bin size      : {rate_seconds}s  |  Output : '{output_dir}'")
    print(f"Mode          : ALL rows kept — timestamps snapped to nearest {rate_seconds}s bin\n")
    print(f"{'#':<5} {'File':<45} {'Rows in':>10} {'Rows out':>10} {'Size':>9}  Status")
    print("-" * 90)

    for idx, src in enumerate(files, 1):
        stem      = src.name.removesuffix(".gz").removesuffix(".csv")
        dst       = output_dir / f"{stem}_{rate_seconds}s.parquet"
        file_date = parse_date_from_stem(stem)

        try:
            df_raw       = read_csv_gz(src, file_date)
            df_resampled = resample(df_raw, rate_seconds)
            size_mb      = export_parquet(df_resampled, dst)

            # rows in == rows out (no data lost)
            assert len(df_raw) == len(df_resampled), "Row count mismatch — this should never happen"

            print(
                f"{idx:<5} {src.name:<45} "
                f"{len(df_raw):>10,} {len(df_resampled):>10,} "
                f"{size_mb:>7.2f}MB  ✓"
            )
            results.append((df_raw, df_resampled, stem))
            ok += 1

        except Exception as exc:
            print(f"{idx:<5} {src.name:<45} {'—':>10} {'—':>10} {'—':>9}  ✗  {exc}")
            failed.append(src.name)

    print("-" * 90)
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
        from shared.plot_lfr import plot_lfr
        for df_raw, df_resampled, stem in results:
            plot_pair(df_raw, df_resampled, stem, stem, NEW_SAMPLING_RATE)
            plot_lfr(df_resampled, stem, NEW_SAMPLING_RATE, raw=df_raw)
        print("Done.")