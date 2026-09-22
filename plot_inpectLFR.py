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
import inspect
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # never opens a window / plot pane — only saves to file
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


# %% Config
# ── CONFIG — edit here and run ──────────────────────────────────────────────
INPUT_DIR           = r"data\raw"
OUTPUT_DIR          = r"data\plots"
FINE_RATE_SECONDS   = 1      # seconds, bin size for the 1st figure (max data resolution)
NEW_SAMPLING_RATE   = 15     # seconds, bin size for the 2nd figure
TO_UTC              = 0      # hours to add to timestamps to convert to UTC (0 = already UTC)
TIME_SLICE           = (0, 24)  # (HI, HF) hora inicial e final do zoom no eixo X, 0-24; (0, 24) = dia inteiro. Só dá zoom na figura — a média móvel continua calculada sobre o dia inteiro.
VERTICAL_OFFSET     = None   # None = auto-computed once from the busiest day; or a fixed number (same unit as the Y axis) for every row
OFFSET_MARGIN       = -0.35  # fraction of the busiest day's peak added to the spacing; negative = adjacent curves intentionally overlap (only used if VERTICAL_OFFSET is None)
ROW_HEIGHT_IN       = 1.4    # inches of height per day row in the figures — raise this for a taller figure
GENERATE_15S_FIGURE = False  # set True to also generate the NEW_SAMPLING_RATE figure alongside the fine (1s) one
SHOW_MOVING_AVERAGE  = True  # overlay a moving-average low-pass curve on the fine (1s) figure
MOVING_AVG_WINDOW_S  = 45    # seconds; -3dB low-pass cutoff ≈ 0.443/window ≈ 7 mHz for 1s-binned data
# --- DESPIKE: burst handling (only when the moving average is shown) ---
DESPIKE_MODE         = "ignore"  # "off" = untouched | "ignore" = bursts skipped by the moving average | "mean" = bursts replaced by the local mean
DESPIKE_FACTOR       = 8.0   # a bin is a burst if count > FACTOR x local mean ...
DESPIKE_MIN_COUNT    = 20    # ... AND count >= this (avoids flagging a few flashes in quiet periods)
DESPIKE_WINDOW_S     = 301   # seconds; window of the local mean used to detect/replace bursts
DESPIKE_CLEAN_RAW    = False # True = the raw curve is also drawn without the bursts (False = raw stays untouched)
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


def save_figure(fig, out_path: Path, retries: int = 3, delay_s: float = 1.0) -> None:
    """
    Save with a few retries: on Windows, saving to a folder that's being
    watched/synced (OneDrive, antivirus, indexing) can transiently fail
    right after another file was just written there. A short pause and
    retry usually clears it without bothering the user.
    """
    for attempt in range(1, retries + 1):
        try:
            fig.savefig(out_path, dpi=150)
            return
        except OSError as exc:
            if attempt == retries:
                raise
            print(f"    Warning: save failed ({exc}), retrying ({attempt}/{retries})...")
            time.sleep(delay_s)


def add_provenance_footer(fig, storage_path, x=0.9, y=0.01, **kwargs):
    """
    Adiciona rodapé com Script / Function / Storage no rodapé da figura,
    descobrindo automaticamente (via inspect) o arquivo e a função que
    chamaram esta rotina — evita rodapés dessincronizados ao copiar/colar
    entre scripts.
    """
    caller = inspect.stack()[1]
    script_name   = os.path.relpath(caller.filename)
    function_name = caller.function
    storage_text  = str(storage_path)

    text = (
        f"Script: {script_name}\n"
        f"Function: {function_name}\n"
        f"Storage: {storage_text}"
    )

    defaults = dict(
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=10,
        color='black',
        alpha=0.7,
    )
    defaults.update(kwargs)

    fig.text(x, y, text, **defaults)


def flag_bursts(counts: pd.Series, rate_seconds: int) -> pd.Series:
    """
    DESPIKE: boolean mask of bins considered bursts — count greater than
    DESPIKE_FACTOR x the local mean (centered window of DESPIKE_WINDOW_S
    seconds) AND at least DESPIKE_MIN_COUNT.
    """
    window_bins = max(3, round(DESPIKE_WINDOW_S / rate_seconds))
    local_mean = counts.rolling(window=window_bins, center=True, min_periods=1).mean()
    return (counts > DESPIKE_FACTOR * local_mean) & (counts >= DESPIKE_MIN_COUNT)


def rolling_mean_convolve(values: pd.Series, window_bins: int) -> pd.Series:
    """
    Centered moving average via numpy.convolve — equivalent to
    values.rolling(window_bins, center=True, min_periods=1).mean(), but
    computed with a single convolution instead of pandas' rolling window
    (much faster for long 1s-binned series).

    NaNs in `values` (e.g. burst bins masked out by DESPIKE_MODE="ignore")
    are excluded from the average rather than propagated: the kernel is
    convolved once against the data (NaNs -> 0) and once against a mask
    of valid bins, and the two sums are divided so each output point is
    the mean of only the valid bins that fall in its window — same as
    pandas' NaN handling. Near the edges the window is naturally smaller
    (fewer bins on one side), like min_periods=1 does.
    """
    arr = values.to_numpy(dtype=float)
    valid = ~np.isnan(arr)
    filled = np.where(valid, arr, 0.0)

    kernel = np.ones(window_bins, dtype=float)
    sums   = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid.astype(float), kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        result = sums / counts
    result[counts == 0] = np.nan  # bin had no valid data anywhere in its window

    return pd.Series(result, index=values.index)


def plot_flash_rate_stack(
    days_data: list[tuple[pd.Timestamp, pd.DataFrame]],
    rate_seconds: int,
    out_path: Path,
    moving_avg_window_s: int | None = None,
    show_raw: bool = True,
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

    Each day gets its own color (sampled from a colormap) and its date
    is shown in the legend. Vertical spacing between rows is FIXED —
    computed once (from the busiest day's peak, scaled by OFFSET_MARGIN,
    or from VERTICAL_OFFSET if set) and applied equally to every row. A
    negative OFFSET_MARGIN intentionally lets adjacent curves overlap.
    """
    if DESPIKE_MODE not in ("off", "ignore", "mean"):
        raise ValueError(f"DESPIKE_MODE must be 'off', 'ignore' or 'mean' (got {DESPIKE_MODE!r})")

    counts_per_day = []
    for date, df in days_data:
        ts_binned = df["time_stamp"].dt.round(f"{rate_seconds}s")
        counts = ts_binned.value_counts().sort_index()

        full_index = pd.date_range(date, date + pd.Timedelta(days=1), freq=f"{rate_seconds}s", inclusive="left")
        counts_full = counts.reindex(full_index, fill_value=0)

        # --- DESPIKE: series feeding the moving average (and optionally the raw curve) ---
        clean = counts_full
        if DESPIKE_MODE != "off" and moving_avg_window_s:
            mask = flag_bursts(counts_full, rate_seconds)
            if DESPIKE_MODE == "ignore":
                clean = counts_full.mask(mask)  # NaN -> rolling mean skips those bins
            elif DESPIKE_MODE == "mean":
                win = max(3, round(DESPIKE_WINDOW_S / rate_seconds))
                local = counts_full.mask(mask).rolling(win, center=True, min_periods=1).mean()
                clean = counts_full.where(~mask, local)
            if show_raw:  # print once per day (the moving-average-only figure reuses the same mask)
                n_flag = int(mask.sum())
                share = counts_full[mask].sum() / max(counts_full.sum(), 1)
                print(f"    {date.date()}: {n_flag:,} burst bins flagged "
                      f"({n_flag / len(counts_full):.2%} of bins, {share:.1%} of events) -> mode '{DESPIKE_MODE}'")

        smoothed = None
        if moving_avg_window_s:
            window_bins = max(1, round(moving_avg_window_s / rate_seconds))
            smoothed = rolling_mean_convolve(clean, window_bins)

        raw_plot = clean if DESPIKE_CLEAN_RAW else counts_full

        counts_per_day.append((date, raw_plot, smoothed))

    cutoff_mhz = 0.443 / moving_avg_window_s * 1000 if moving_avg_window_s else None

    fig, ax = plt.subplots(figsize=(14, ROW_HEIGHT_IN * len(days_data) + 2))

    paired = plt.cm.tab20_r.colors  # 12 colors = 6 raw/smoothed pairs (cycles if > 6 days)
    raw_colors      = [paired[(2 * i) % len(paired)] for i in range(len(days_data))]
    smoothed_colors = [paired[(2 * i + 1) % len(paired)] for i in range(len(days_data))]
    


    if show_raw:
        global_max = max((c.max() if len(c) else 0 for _, c, _ in counts_per_day), default=0)
        fixed_step = VERTICAL_OFFSET if VERTICAL_OFFSET else max(global_max * (1 + OFFSET_MARGIN), 1.0)
    else:
        fixed_step = 0.0

    y_base = 0.0
    for idx, (date, counts_full, smoothed) in enumerate(counts_per_day):
        raw_color = raw_colors[idx]
        smoothed_color = smoothed_colors[idx]
        ax.axhline(y_base, color="gray", linewidth=0.5, alpha=0.3, zorder=1)
        if len(counts_full):
            x = hours_of_day(counts_full.index.to_series(), date)
            date_label = date.strftime("%Y-%m-%d")
            if smoothed is not None:
                if show_raw:
                    ax.scatter(x, counts_full.values + y_base, s=2, color=raw_color, zorder=0 + idx)
                ax.plot(x, smoothed.values + y_base, linewidth=1.0, color=smoothed_color, label=date_label, zorder=100 + idx)
            else:
                ax.plot(x, counts_full.values + y_base, linewidth=1.1, color=raw_color, label=date_label, zorder=2 + idx)
    
        y_base += fixed_step

    ax.legend(loc="upper right", fontsize=8, ncol=max(1, len(days_data) // 15))

    x_start, x_end = TIME_SLICE
    ax.set_xlim(x_start, x_end)
    ax.set_xlabel("Time of day (UTC)")
    major_step = 3 if (x_end - x_start) > 6 else 1  # finer ticks when zoomed into a short window
    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(MultipleLocator(1 if major_step > 1 else 0.25))
    ax.tick_params(axis="x", which="minor", length=3)
    ax.tick_params(axis="x", which="major", length=6)
    ax.grid(axis="x", which="major", linewidth=0.6, alpha=0.4)
    ax.grid(axis="x", which="minor", linewidth=0.4, alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_yticks([])
    if rate_seconds == 1:
        ax.set_ylabel("Lightning flash rate (flashes/sec)")
    else:
        ax.set_ylabel(f"Lightning flash rate (flashes/{rate_seconds}s)")
    title = f"Flash rate — resampled, {rate_seconds}s bins (rounded)"
    if moving_avg_window_s:
        # title += f"  +  ~{cutoff_mhz:.1f} mHz low-pass"
        title += f"  +  ~{MOVING_AVG_WINDOW_S} sec mov. avg"
        if DESPIKE_MODE != "off":
            title += f"  |  bursts >{DESPIKE_FACTOR:g}x local mean: {DESPIKE_MODE}"
            title += " (raw too)" if DESPIKE_CLEAN_RAW else ""
    ax.set_title(title)
    # reserva ~0.75" na base para o rodapé (em fração da altura, já que a figura
    # muda de altura conforme o nº de dias) e evita que o tight_layout o cubra
    footer_frac = 0.75 / fig.get_figheight()
    fig.tight_layout(rect=[0, footer_frac, 1, 1])
    add_provenance_footer(fig, out_path)
    save_figure(fig, out_path, dpi = 300)
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
    ma_out     = output_dir / f"lfr_moving_avg_{MOVING_AVG_WINDOW_S}s_{date_tag}.png"
    coarse_out = output_dir / f"lfr_resampled_{NEW_SAMPLING_RATE}s_{date_tag}.png"

    print(f"\nGenerating figure with {FINE_RATE_SECONDS}s bins...")
    plot_flash_rate_stack(
        days_data, FINE_RATE_SECONDS, fine_out,
        moving_avg_window_s=MOVING_AVG_WINDOW_S if SHOW_MOVING_AVERAGE else None,
    )
    print(f"  saved -> {fine_out}")

    if SHOW_MOVING_AVERAGE:
        print("Generating moving-average-only figure...")
        plot_flash_rate_stack(
            days_data, FINE_RATE_SECONDS, ma_out,
            moving_avg_window_s=MOVING_AVG_WINDOW_S, show_raw=False,
        )
        print(f"  saved -> {ma_out}")

    if GENERATE_15S_FIGURE:
        print(f"Generating figure with {NEW_SAMPLING_RATE}s bins...")
        plot_flash_rate_stack(days_data, NEW_SAMPLING_RATE, coarse_out)
        print(f"  saved -> {coarse_out}")
    else:
        print(f"{NEW_SAMPLING_RATE}s figure skipped (GENERATE_15S_FIGURE = False)")

    print("\nDone.")


if __name__ == "__main__":
    main()