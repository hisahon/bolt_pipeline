"""
shared/plot_lfr.py

Lightning Flash Rate (LFR) line plot for the ingest_lightning pipeline.
Called automatically when VISUALIZATION_PLOTS = True in ingest_lightning.py,
or run standalone by setting the CONFIG block at the bottom.

Produces one PNG per file in data/images/:
  stem_lfr.png  — LFR line plot in flashes/s over the full day

Both series share the same physical unit: flashes/s.
Each bin count is divided by its duration so the y-axis is
directly comparable regardless of bin size.
"""

import matplotlib
matplotlib.use("Agg")  # save to file only — no Spyder panel popup

import matplotlib as mpl
mpl.rcParams["font.size"]       = 11
mpl.rcParams["axes.titlesize"]  = 13
mpl.rcParams["axes.labelsize"]  = 11
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9

import gzip
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PLOT_DIR = Path("data") / "images"


# ── public API ────────────────────────────────────────────────────────────────
def plot_lfr(resampled: pd.DataFrame,
             stem: str,
             sampling_rate: int = 15,
             raw: pd.DataFrame | None = None) -> None:
    """
    Generate and save a Lightning Flash Rate (LFR) line plot.

    Parameters
    ----------
    resampled     : binned DataFrame produced by ingest_lightning.py
                    (time_stamp already snapped to nearest sampling_rate seconds)
    stem          : base filename (no extension) used for the output PNG
    sampling_rate : bin size in seconds — used for axis labels and title
    raw           : optional original DataFrame with millisecond timestamps;
                    when provided, LFR is computed at 1s resolution and
                    plotted alongside for comparison
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # full-day x-axis anchors
    day_start = resampled["time_stamp"].dt.normalize().iloc[0]
    day_end   = day_start + pd.Timedelta(hours=24)
    date      = day_start.date()
    doy       = day_start.day_of_year
    total     = len(resampled)

    # ── LFR resampled: flashes per bin / bin_duration → flashes/s ────────────
    lfr_res = (
        resampled.groupby("time_stamp")
                 .size()
                 .rename("lfr")
                 / sampling_rate          # convert flashes/Ns → flashes/s
    ).reset_index().sort_values("time_stamp")

    # ── LFR raw (optional): 1s bins / 1s → flashes/s ─────────────────────────
    lfr_raw = None
    if raw is not None:
        lfr_raw = (
            raw.set_index("time_stamp")
               .resample("1s")
               .size()
               .rename("lfr")
               / 1                        # bin = 1s, already flashes/s
        ).reset_index().sort_values("time_stamp")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))

    if lfr_raw is not None:
        ax.plot(
            lfr_raw["time_stamp"],
            lfr_raw["lfr"],
            color="#EF9F27",
            linewidth=0.6,
            alpha=0.7,
            label="Original (1s bins)",
        )

    ax.plot(
        lfr_res["time_stamp"],
        lfr_res["lfr"],
        color="#1D9E75",
        linewidth=1.2,
        label=f"Resampled ({sampling_rate}s bins)",
    )

    n_bins = len(lfr_res)
    ax.set_title(
        f"{date} (DOY {doy})  —  LFR @ {sampling_rate}s  "
        f"|  {total:,} flashes  |  {n_bins:,} bins"
    )
    ax.set_ylabel("LFR  (flashes / s)")
    ax.set_xlabel("Hour (UTC)")

    ax.set_xlim(day_start, day_end)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(mpl.dates.HourLocator(byhour=range(0, 24, 3)))
    ax.xaxis.set_minor_locator(mpl.dates.HourLocator(byhour=range(0, 24, 1)))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H"))

    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.4)
    
    fig.text(0.9, -0.01, 'Script: bolt-pipeline/shared/plot_lfr.py\nFunction: plot_lfr\nStorage: bolt_pipeline/data/images', 
             horizontalalignment='right', 
             verticalalignment='bottom', 
             fontsize=10, 
             color='red', 
             alpha=0.7)

    fig.tight_layout()

    out = PLOT_DIR / f"{stem}_lfr.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {out}")


# ── standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── STANDALONE CONFIG — only used when running this file directly ─────────
    OUTPUT_DIR        = r"data\parquet_15s"
    NEW_SAMPLING_RATE = 15
    MAX_FILES         = 3
    # ─────────────────────────────────────────────────────────────────────────

    par_files = sorted(Path(OUTPUT_DIR).glob(f"*_{NEW_SAMPLING_RATE}s.parquet"))[:MAX_FILES]

    if not par_files:
        print(f"No *_{NEW_SAMPLING_RATE}s.parquet files found in: {OUTPUT_DIR}")
    else:
        for par_path in par_files:
            stem = par_path.stem
            print(f"\nLoading {par_path.name} ...")
            resampled = pd.read_parquet(par_path)
            plot_lfr(resampled, stem, NEW_SAMPLING_RATE)

        print(f"\nAll LFR plots saved to: {PLOT_DIR}")