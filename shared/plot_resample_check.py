"""
shared/plot_resample_check.py

QA plots for the csv_gz_to_parquet conversion pipeline.
Called automatically when VISUALIZATION_PLOTS = True in csv_gz_to_parquet.py,
or run standalone by setting the CONFIG block below.

Produces one PNG per file in data/images/:
  stem_overview.png  — time series + side-by-side lon x lat scatter (coloured by hour)
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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

PLOT_DIR = Path("data") / "images"
CMAP     = "plasma"


# ── internal helpers ──────────────────────────────────────────────────────────
def _add_hour_decimal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_decimal"] = df["time_stamp"].dt.hour + df["time_stamp"].dt.minute / 60
    return df


def _scatter_hour(ax, df, title, norm, s=1, alpha=0.4):
    ax.scatter(
        df["longitude"], df["latitude"],
        c=df["hour_decimal"], cmap=CMAP, norm=norm,
        s=s, alpha=alpha, linewidths=0, rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def _shared_limits(raw, resampled):
    x_lim = (
        min(raw["longitude"].min(), resampled["longitude"].min()) - 0.5,
        max(raw["longitude"].max(), resampled["longitude"].max()) + 0.5,
    )
    y_lim = (
        min(raw["latitude"].min(), resampled["latitude"].min()) - 0.5,
        max(raw["latitude"].max(), resampled["latitude"].max()) + 0.5,
    )
    return x_lim, y_lim


# ── public API ────────────────────────────────────────────────────────────────
# def plot_pair(raw: pd.DataFrame, resampled: pd.DataFrame,
#               label: str, stem: str, sampling_rate: int = 15) -> None:
#     """
#     Generate and save the overview QA figure for one file pair.

#     Parameters
#     ----------
#     raw           : original DataFrame (from csv.gz)
#     resampled     : resampled DataFrame (from parquet)
#     label         : title string shown on figure
#     stem          : base filename (no extension) used for output PNG
#     sampling_rate : rate in seconds, used in titles
#     """
#     PLOT_DIR.mkdir(parents=True, exist_ok=True)

#     raw       = _add_hour_decimal(raw)
#     resampled = _add_hour_decimal(resampled)

#     n_raw = len(raw)
#     n_res = len(resampled)
#     kept  = 100 * n_res / n_raw if n_raw else 0
#     norm  = Normalize(vmin=0, vmax=24)
#     x_lim, y_lim = _shared_limits(raw, resampled)

#     fig = plt.figure(figsize=(16, 14))
#     # fig.suptitle(
#     #     f"{label}  —  "
#     #     f"Original: {n_raw:,}  →  Resampled @{sampling_rate}s: {n_res:,}  ({kept:.1f}% kept)",
#     #     fontweight="bold", y=0.99,
#     # )

#     gs = gridspec.GridSpec(
#         2, 1, figure=fig,
#         hspace=0.08,
#         height_ratios=[1, 3],
#     )

#     # ── time series ───────────────────────────────────────────────────────────
#     ax_ts = fig.add_subplot(gs[0])
#     raw_counts = raw.set_index("time_stamp").resample("1min").size()
#     res_counts = resampled.set_index("time_stamp").resample("1min").size()
#     ax_ts.plot(raw_counts.index, raw_counts.values,
#                color="#EF9F27", linewidth=0.9, label="Original (1 min)")
#     ax_ts.plot(res_counts.index, res_counts.values,
#                color="#1D9E75", linewidth=0.9,
#                label=f"Resampled @{sampling_rate}s (1 min)")
#     doy = raw["time_stamp"].dt.dayofyear.iloc[0]
#     date = raw["time_stamp"].dt.date.iloc[0]
#     ax_ts.set_title(f"{date} (DOY {doy}) — Strike count per minute — Resampled @{sampling_rate}s: {n_res:,} ({kept:.1f}% kept)")
#     ax_ts.set_ylabel("Strikes / min")
#     ax_ts.legend()
#     ax_ts.tick_params(axis="x", labelrotation=0)
#     ax_ts.xaxis.set_major_locator(mpl.dates.HourLocator(byhour=range(0, 24, 3)))
#     ax_ts.xaxis.set_minor_locator(mpl.dates.HourLocator(byhour=range(0, 24, 1)))
#     ax_ts.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H"))
#     ax_ts.set_xlabel("Hour (UTC)")
#     day_start = raw["time_stamp"].dt.normalize().iloc[0]
#     day_end   = day_start + pd.Timedelta(hours=24)
#     ax_ts.set_xlim(day_start, day_end)

#     # ── side-by-side scatters (equal physical size via ImageGrid) ─────────────
#     scatter_area = fig.add_subplot(gs[1])
#     scatter_area.set_visible(False)   # placeholder to anchor ImageGrid position

#     grid = ImageGrid(
#         fig,
#         gs[1],
#         nrows_ncols=(1, 2),
#         axes_pad=0.4,
#         share_all=True,
#         cbar_location="right",
#         cbar_mode="single",
#         cbar_size="3%",
#         cbar_pad=0.1,
#     )

#     _scatter_hour(grid[0], raw,
#                   f"Original — {n_raw:,} strikes",
#                   norm, s=1, alpha=0.2)
#     grid[0].set_xlim(x_lim)
#     grid[0].set_ylim(y_lim)

#     _scatter_hour(grid[1], resampled,
#                   f"Resampled @{sampling_rate}s — {n_res:,} strikes",
#                   norm, s=2, alpha=0.4)
#     grid[1].set_xlim(x_lim)
#     grid[1].set_ylim(y_lim)

#     cbar = grid.cbar_axes[0].colorbar(
#         ScalarMappable(norm=norm, cmap=CMAP)
#     )
#     cbar.set_label("Hour of day (decimal)")
#     cbar.set_ticks(range(0, 25, 3))

#     out = PLOT_DIR / f"{stem}_overview.png"
#     fig.savefig(out, dpi=130, bbox_inches="tight")
#     plt.close(fig)
#     print(f"   Saved → {out}")

# 

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# from mpl_toolkits.axes_grid1 import ImageGrid
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import pandas as pd


def _add_map_features(ax) -> None:
    """Adiciona costa, fronteiras de país e de estado ao eixo cartopy."""
    ax.coastlines(resolution="50m", linewidth=0.6, color="black", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="black", zorder=3)
    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="50m",
        facecolor="none",
        edgecolor="gray",
    )
    ax.add_feature(states, linewidth=0.5, zorder=3)


def _add_gridlines(ax, left_labels: bool = True, bottom_labels: bool = True) -> None:
    """Adiciona gridlines com labels de lat/lon."""
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left_labels
    gl.bottom_labels = bottom_labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


def plot_pair(raw: pd.DataFrame, resampled: pd.DataFrame,
              label: str, stem: str, sampling_rate: int = 15) -> None:
    """
    Generate and save the overview QA figure for one file pair.
    Parameters
    ----------
    raw           : original DataFrame (from csv.gz)
    resampled     : resampled DataFrame (from parquet)
    label         : title string shown on figure
    stem          : base filename (no extension) used for output PNG
    sampling_rate : rate in seconds, used in titles
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    raw       = _add_hour_decimal(raw)
    resampled = _add_hour_decimal(resampled)
    n_raw = len(raw)
    n_res = len(resampled)
    kept  = 100 * n_res / n_raw if n_raw else 0
    norm  = Normalize(vmin=0, vmax=24)
    x_lim, y_lim = _shared_limits(raw, resampled)
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        hspace=0.08,
        height_ratios=[1, 3],
    )
    # ── time series ───────────────────────────────────────────────────────────
    ax_ts = fig.add_subplot(gs[0])
    raw_counts = raw.set_index("time_stamp").resample("1min").size()
    res_counts = resampled.set_index("time_stamp").resample("1min").size()
    ax_ts.plot(raw_counts.index, raw_counts.values,
               color="#EF9F27", linewidth=0.9, label="Original (1 min)")
    ax_ts.plot(res_counts.index, res_counts.values,
               color="#1D9E75", linewidth=0.9,
               label=f"Resampled @{sampling_rate}s (1 min)")
    doy = raw["time_stamp"].dt.dayofyear.iloc[0]
    date = raw["time_stamp"].dt.date.iloc[0]
    ax_ts.set_title(f"{date} (DOY {doy}) — Strike count per minute — Resampled @{sampling_rate}s: {n_res:,} ({kept:.1f}% kept)")
    ax_ts.set_ylabel("Strikes / min")
    ax_ts.legend()
    ax_ts.tick_params(axis="x", labelrotation=0)
    ax_ts.xaxis.set_major_locator(mpl.dates.HourLocator(byhour=range(0, 24, 3)))
    ax_ts.xaxis.set_minor_locator(mpl.dates.HourLocator(byhour=range(0, 24, 1)))
    ax_ts.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H"))
    ax_ts.set_xlabel("Hour (UTC)")
    day_start = raw["time_stamp"].dt.normalize().iloc[0]
    day_end   = day_start + pd.Timedelta(hours=24)
    ax_ts.set_xlim(day_start, day_end)
    # ── side-by-side scatters (equal physical size via ImageGrid) ─────────────
    scatter_area = fig.add_subplot(gs[1])
    scatter_area.set_visible(False)   # placeholder to anchor ImageGrid position
    proj = ccrs.PlateCarree()
    grid = ImageGrid(
        fig,
        gs[1],
        nrows_ncols=(1, 2),
        axes_pad=0.6,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="3%",
        cbar_pad=0.1,
        axes_class=(GeoAxes, dict(projection=proj)),
    )
    _scatter_hour(grid[0], raw,
                  f"Original — {n_raw:,} strikes",
                  norm, s=1, alpha=0.2)
    _add_map_features(grid[0])
    _add_gridlines(grid[0], left_labels=True, bottom_labels=True)
    grid[0].set_extent([x_lim[0], x_lim[1], y_lim[0], y_lim[1]], crs=proj)
    _scatter_hour(grid[1], resampled,
                  f"Resampled @{sampling_rate}s — {n_res:,} strikes",
                  norm, s=2, alpha=0.4)
    _add_map_features(grid[1])
    _add_gridlines(grid[1], left_labels=False, bottom_labels=True)
    grid[1].set_extent([x_lim[0], x_lim[1], y_lim[0], y_lim[1]], crs=proj)
    cbar = grid.cbar_axes[0].colorbar(
        ScalarMappable(norm=norm, cmap=CMAP)
    )
    cbar.set_label("Hour of day (decimal)")
    cbar.set_ticks(range(0, 25, 3))
    out = PLOT_DIR / f"{stem}_overview.png"
    
    fig.text(0.9, 0.01, 'Script: bolt-pipeline/shared/plot_resample_check.py\nFunction: plot_pair\nStorage: bolt_pipeline/data/images', 
             horizontalalignment='right', 
             verticalalignment='bottom', 
             fontsize=10, 
             color='red', 
             alpha=0.7)
    
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {out}")


# ── standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── STANDALONE CONFIG — only used when running this file directly ─────────
    INPUT_DIR         = r"data\raw"
    OUTPUT_DIR        = r"data\parquet_15s"
    NEW_SAMPLING_RATE = 15
    MAX_FILES         = 3
    # ─────────────────────────────────────────────────────────────────────────

    def _load_raw(path: Path) -> pd.DataFrame:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            df = pd.read_csv(
                f,
                usecols=["latitude", "longitude", "time_stamp", "type_data"],
                dtype={"latitude": "float64", "longitude": "float64",
                       "time_stamp": "str", "type_data": "str"},
            )
        df["time_stamp"] = pd.to_datetime(
            df["time_stamp"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
        )
        return df.dropna(subset=["time_stamp"])

    raw_files = sorted(Path(INPUT_DIR).glob("*.csv.gz"))[:MAX_FILES]

    if not raw_files:
        print(f"No .csv.gz files found in: {INPUT_DIR}")
    else:
        for src in raw_files:
            stem     = src.name.removesuffix(".gz").removesuffix(".csv")
            par_path = Path(OUTPUT_DIR) / f"{stem}_{NEW_SAMPLING_RATE}s.parquet"

            if not par_path.exists():
                print(f"Skipping {src.name} — parquet not found: {par_path.name}")
                continue

            print(f"\nLoading {src.name} ...")
            raw       = _load_raw(src)
            resampled = pd.read_parquet(par_path)

            plot_pair(raw, resampled, stem, stem, NEW_SAMPLING_RATE)

        print(f"\nAll plots saved to: {PLOT_DIR}")