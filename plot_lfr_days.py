"""
CLI to compare lightning activity (LFR) across days.

The list of available days comes only from the file NAMES (fast, doesn't
open any file) — same convention as ingest_lightning.py:
lx_raw_ROGER_20220407_entln -> 2022-04-07. You choose which days you want,
and ONLY THEN are the corresponding files read (never reads a file for a
day you didn't ask for).

A file may contain data beyond the day indicated in its name (leftover
rows from the previous/next day, for example) — any row outside the exact
[00:00:00.000, 24:00:00.000) interval of that day is discarded before
plotting.

For the chosen days, generates TWO flash-rate (LFR) figures — each day in
its own row, with a vertical offset, X axis = hour of day (UTC), 0-24h.
Both figures use exactly the same process (round each timestamp to the
nearest multiple of N seconds via pandas .dt.round, count how many
flashes fall in each bin) — only the value of N changes:

    1) lfr_resampled_{FINE_RATE_SECONDS}s_<...>.png
       maximum resolution — bins of FINE_RATE_SECONDS seconds (default: 1s).

    2) lfr_resampled_{NEW_SAMPLING_RATE}s_<...>.png
       bins of NEW_SAMPLING_RATE seconds (default: 15s).

Figures are only saved to OUTPUT_DIR (backend "Agg", no plt.show()) —
runs fine from a terminal, no need for Spyder's plot pane.

Usage:
    python plot_lfr_days.py
"""

# %% Imports
import gzip
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # never opens a window / plot pane — only saves to file
import matplotlib.pyplot as plt
import pandas as pd


# %% Config
# ── CONFIG — edit here and run ──────────────────────────────────────────────
INPUT_DIR           = r"data\raw"
OUTPUT_DIR          = r"data\plots"
FINE_RATE_SECONDS   = 1      # seconds, bin size for the 1st figure (max data resolution)
NEW_SAMPLING_RATE   = 15     # seconds, bin size for the 2nd figure
TO_UTC              = 0      # hours to add to timestamps to convert to UTC (0 = already UTC)
VERTICAL_OFFSET     = None   # None = auto-computed once from the busiest day; or a fixed number (same unit as the Y axis) for every row
OFFSET_MARGIN       = 0.15   # headroom above the busiest day's curve, as a fraction of its peak (only used if VERTICAL_OFFSET is None) — applied equally to every row
ROW_HEIGHT_IN       = 0.6    # inches of height per day row in the figures — lower this to make everything more compact
GENERATE_15S_FIGURE = True   # set False to skip the NEW_SAMPLING_RATE figure while iterating on the fine one
SHOW_MOVING_AVERAGE  = True  # overlay a moving-average low-pass curve on the fine (1s) figure
MOVING_AVG_WINDOW_S  = 63    # seconds; -3dB low-pass cutoff ≈ 0.443/window ≈ 7 mHz for 1s-binned data
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "latitude":   "float64",
    "longitude":  "float64",
    "time_stamp": "str",
    "type_data":  "Int8",
}
KEEP_COLUMNS = ["latitude", "longitude", "time_stamp", "type_data"]


# %% Functions
def parse_date_from_stem(stem: str) -> pd.Timestamp | None:
    """Extract the date from the filename, e.g. lx_raw_ROGER_20220407_entln -> 2022-04-07."""
    match = re.search(r"(\d{8})", stem)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d")
    return None


def discover_files(input_dir: Path) -> list[tuple[pd.Timestamp, Path]]:
    """List the .csv.gz files and pair each with the date parsed from its name — opens no files."""
    entries = []
    for path in sorted(input_dir.glob("*.csv.gz")):
        stem = path.name.removesuffix(".gz").removesuffix(".csv")
        date = parse_date_from_stem(stem)
        if date is None:
            print(f"  (skipped, no date in name: {path.name})")
            continue
        entries.append((date, path))
    entries.sort(key=lambda e: e[0])
    return entries


def prompt_day_selection(entries: list[tuple[pd.Timestamp, Path]]) -> list[tuple[pd.Timestamp, Path]]:
    """Show the numbered list of days and read the user's choice from the terminal."""
    print(f"\nAvailable days ({len(entries)}):\n")
    for i, (date, path) in enumerate(entries, 1):
        print(f"  [{i:>3}] {date.date()}   ({path.name})")

    print("\nEnter the numbers of the days you want to plot.")
    print("Examples: '1,3,5'  or  '1-4'  or  '1-3,7'  or 'all'\n")

    raw = input("Days > ").strip()
    if raw.lower() == "all":
        return entries

    chosen_idx = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-")
            chosen_idx.update(range(int(a), int(b) + 1))
        else:
            chosen_idx.add(int(chunk))

    selected = [entries[i - 1] for i in sorted(chosen_idx) if 1 <= i <= len(entries)]
    if not selected:
        print("No valid days selected. Exiting.")
        sys.exit(0)
    return selected


def load_day(path: Path, date: pd.Timestamp) -> pd.DataFrame:
    """
    Read the file for the selected day and keep only the rows within
    [date 00:00:00.000, date+1 00:00:00.000) — any extra data the file
    might have (from another day) is discarded here, before plotting.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, dtype=DTYPE_MAP, usecols=KEEP_COLUMNS)

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed", errors="coerce")
    nat_count = df["time_stamp"].isna().sum()
    if nat_count:
        print(f"    Warning: {nat_count:,} rows with invalid timestamps dropped")
        df = df.dropna(subset=["time_stamp"])

    df["time_stamp"] = df["time_stamp"].dt.tz_localize(None)
    if TO_UTC:
        df["time_stamp"] = df["time_stamp"] + pd.Timedelta(hours=TO_UTC)

    day_start = date
    day_end   = date + pd.Timedelta(days=1)
    before    = len(df)
    df        = df[(df["time_stamp"] >= day_start) & (df["time_stamp"] < day_end)].copy()
    dropped   = before - len(df)
    if dropped:
        print(f"    Info: {dropped:,} rows outside {date.date()} discarded (didn't belong to this day)")

    return df.reset_index(drop=True)


def hours_of_day(ts: pd.Series, midnight: pd.Timestamp) -> pd.Series:
    """Convert a series of timestamps to 'hour of day' (0-24), relative to midnight."""
    return (ts - midnight).dt.total_seconds() / 3600.0


def plot_flash_rate_stack(
    days_data: list[tuple[pd.Timestamp, pd.DataFrame]],
    rate_seconds: int,
    out_path: Path,
    moving_avg_window_s: int | None = None,
) -> None:
    """
    Flash rate per rate_seconds-second bin: each timestamp is rounded
    (pandas .dt.round) to the nearest multiple of rate_seconds, and we
    count how many flashes fall in each bin. Same process regardless of
    rate_seconds — used both with 1s (max resolution) and any other value
    (e.g. 15s).

    Counts are reindexed onto a continuous per-bin grid covering the full
    day, filling seconds/bins with no flashes as 0 (instead of leaving
    them absent from the index) — otherwise matplotlib would draw a
    straight line across quiet gaps instead of showing them as zero, and
    a moving average over a gappy index would be meaningless.

    If moving_avg_window_s is given, a centered moving average (low-pass)
    of that window is overlaid on top of the raw curve, not subtracted.

    Vertical spacing between rows is FIXED — computed once (from the
    busiest day's peak + margin, or from VERTICAL_OFFSET if set) and
    applied equally to every row, so all rows are evenly spaced.
    """
    counts_per_day = []
    for date, df in days_data:
        ts_binned = df["time_stamp"].dt.round(f"{rate_seconds}s")
        counts = ts_binned.value_counts().sort_index()

        full_index = pd.date_range(date, date + pd.Timedelta(days=1), freq=f"{rate_seconds}s", inclusive="left")
        counts_full = counts.reindex(full_index, fill_value=0)

        smoothed = None
        if moving_avg_window_s:
            window_bins = max(1, round(moving_avg_window_s / rate_seconds))
            smoothed = counts_full.rolling(window=window_bins, center=True, min_periods=1).mean()

        counts_per_day.append((date, counts_full, smoothed))

    cutoff_mhz = 0.443 / moving_avg_window_s * 1000 if moving_avg_window_s else None

    fig, ax = plt.subplots(figsize=(14, ROW_HEIGHT_IN * len(days_data) + 2))

    global_max = max((c.max() if len(c) else 0 for _, c, _ in counts_per_day), default=0)
    fixed_step = VERTICAL_OFFSET if VERTICAL_OFFSET else max(global_max * (1 + OFFSET_MARGIN), 1.0)

    y_base = 0.0
    for idx, (date, counts_full, smoothed) in enumerate(counts_per_day):
        ax.axhline(y_base, color="gray", linewidth=0.5, alpha=0.3)
        if len(counts_full):
            x = hours_of_day(counts_full.index.to_series(), date)
            ax.plot(
                x, counts_full.values + y_base, linewidth=0.6, alpha=0.4, color="tab:blue",
                label="raw" if idx == 0 else None,
            )
            if smoothed is not None:
                ax.plot(
                    x, smoothed.values + y_base, linewidth=1.4, color="tab:red",
                    label=f"~{cutoff_mhz:.1f} mHz low-pass" if idx == 0 else None,
                )
        ax.text(-0.3, y_base, str(date.date()), ha="right", va="bottom", fontsize=9)

        y_base += fixed_step

    if moving_avg_window_s:
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlim(0, 24)
    ax.set_xlabel("Time of day (UTC)")
    ax.set_yticks([])
    title = f"Flash rate — resampled, {rate_seconds}s bins (rounded)"
    if moving_avg_window_s:
        title += f"  +  ~{cutoff_mhz:.1f} mHz low-pass"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# %% CLI entry point
def main() -> None:
    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a valid directory.")
        sys.exit(1)

    entries = discover_files(input_dir)
    if not entries:
        print(f"No .csv.gz files with a recognizable date found in: {input_dir}")
        sys.exit(0)

    selected = prompt_day_selection(entries)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {len(selected)} selected day(s)...")
    days_data = []
    for date, path in selected:
        print(f"  {date.date()}  <-  {path.name}")
        df = load_day(path, date)
        if df.empty:
            print(f"    Warning: no valid data for {date.date()}, day included empty")
        days_data.append((date, df))

    days_data.sort(key=lambda d: d[0])

    if len(days_data) > 1:
        date_tag = f"{days_data[0][0].date()}_to_{days_data[-1][0].date()}"
    else:
        date_tag = str(days_data[0][0].date())

    fine_out   = output_dir / f"lfr_resampled_{FINE_RATE_SECONDS}s_{date_tag}.png"
    coarse_out = output_dir / f"lfr_resampled_{NEW_SAMPLING_RATE}s_{date_tag}.png"

    print(f"\nGenerating figure with {FINE_RATE_SECONDS}s bins...")
    plot_flash_rate_stack(
        days_data, FINE_RATE_SECONDS, fine_out,
        moving_avg_window_s=MOVING_AVG_WINDOW_S if SHOW_MOVING_AVERAGE else None,
    )
    print(f"  saved -> {fine_out}")

    if GENERATE_15S_FIGURE:
        print(f"Generating figure with {NEW_SAMPLING_RATE}s bins...")
        plot_flash_rate_stack(days_data, NEW_SAMPLING_RATE, coarse_out)
        print(f"  saved -> {coarse_out}")
    else:
        print(f"{NEW_SAMPLING_RATE}s figure skipped (GENERATE_15S_FIGURE = False)")

    print("\nDone.")


if __name__ == "__main__":
    main()