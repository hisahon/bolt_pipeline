"""
lfr_explorer.py

Interactive Lightning Flash Rate (LFR) explorer for Spyder.

Layout
------
┌─────────────────────────────────────────────────────────────┐
│  ◀ Prev        lx_raw_ROGER_20220407  [1/30]       Next ▶  │
├──────────────────────────┬──────────────────────────────────┤
│                          │  LFR (flashes/s)  [coloured]    │
│   scatter  (square)      ├──────────────────────────────────┤
│   coloured by hour       │  Wavelet power spectrum          │
├──────────────────────────┤  (Torrence & Compo, Morlet)      │
│   colorbar  0h ── 24h    │                                  │
└──────────────────────────┴──────────────────────────────────┘

Usage
-----
1. Set PARQUET_DIR to the folder containing your *_15s.parquet files.
2. Run in Spyder — Graphics backend must NOT be "Inline".
   Tools → Preferences → IPython console → Graphics → Backend → Automatic
3. Draw a rectangle on the scatter map to select a region.
   LFR and wavelet panels update automatically on mouse release.
4. Click "Reset region" or press R to go back to full domain.

Keyboard shortcuts
------------------
  Left  / Right  →  previous / next day
  R              →  reset region

Wavelet parameters (match f2_spec.py exactly)
----------------------------------------------
  dt     = 0.25 h  (time axis in decimal hours, 15 min bins)
  pad=1, dj=0.15, s0=0.90, j1=24, lag1=0.71, mother='MORLET'
  freq slice : 2–10 mHz
  power : normalised 0-1, then log10(p + 0.05)
  display : contourf, cmap='nipy_spectral', levels=20
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib

# pick the best available interactive backend (Qt5 → Qt6 → Tk → WX)
def _set_interactive_backend():
    for backend in ("Qt5Agg", "Qt6Agg", "TkAgg", "WXAgg"):
        try:
            matplotlib.use(backend)
            return backend
        except Exception:
            continue
    raise RuntimeError("No interactive Matplotlib backend found.")

_backend = _set_interactive_backend()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector
from matplotlib.collections import LineCollection

from waveletFunctions import wavelet_on_data_new as wd
from waveletFunctions import wd_slicer


# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_DIR   = r"data\parquet_15s"
SAMPLING_RATE = 15          # seconds — must match ingest_lightning
SCATTER_ALPHA = 0.25
SCATTER_SIZE  = 1
CMAP          = "rainbow"

# domain bounding box — data outside this is discarded at load time
LON_MIN, LON_MAX = -68, -50
LAT_MIN, LAT_MAX = -35, -20

# time range — full day
HOUR_MIN, HOUR_MAX = 15, 22

# wavelet — exact parameters from f2_spec.py
WAV_DT    = 0.25    # time unit = hours; 15-min bins → dt = 0.25 h
WAV_PAD   = 1
WAV_DJ    = 0.15
WAV_S0    = 0.90
WAV_J1    = 24
WAV_LAG1  = 0.71
WAV_MOTHER = "MORLET"
WAV_FREQ_BOUNDS = [2, 10]   # mHz — passed to wd_slicer

# detrend — 5 min window at 4 samples/min = 20 samples
DETREND_WIN = 20
# ─────────────────────────────────────────────────────────────────────────────


# ── colours ───────────────────────────────────────────────────────────────────
BG_DARK  = "#F5F5F5"
BG_PANEL = "#FFFFFF"
C_ACCENT = "#AAAAAA"
C_REGION = "#E05C5C"
C_FULL   = "#1D9E75"


# ── signal helpers ────────────────────────────────────────────────────────────
def _detrend(series: np.ndarray, wins: int) -> np.ndarray:
    """
    Remove running mean — replicates cv.pad_convolve_trim2 exactly.
    Edges within wins//2 are zeroed.
    """
    trimm    = wins // 2
    smoothed = np.convolve(series, np.ones(wins) / wins, mode="same")
    out      = series - smoothed
    out[:trimm]  = 0.0
    out[-trimm:] = 0.0
    return out


def _compute_wavelet(lfr_values: np.ndarray):
    """
    Compute Morlet wavelet power spectrum of LFR.

    Matches f2_spec.py exactly:
      - input  : native 15s LFR array (no re-aggregation)
      - dt=0.25, pad=1, dj=0.15, s0=0.90, j1=24, lag1=0.71
      - x-axis : tms / 24  (fraction of day, 0..1)
      - power  : normalised 0-1, then log10(p + 0.05)

    Returns (tms_day, freq, power) or None if insufficient data.
    """
    signal = _detrend(lfr_values.astype(float), DETREND_WIN)

    if np.count_nonzero(signal) < 16:
        return None

    # wavelet_on_data_new returns (period, freq_mHz, power)
    _period, freq, power = wd(
        signal,
        dt=WAV_DT, pad=WAV_PAD, dj=WAV_DJ,
        s0=WAV_S0, j1=WAV_J1, lag1=WAV_LAG1,
        mother=WAV_MOTHER,
    )

    freq, power = wd_slicer(freq, power, WAV_FREQ_BOUNDS)

    # normalise then log10 — identical to f2_spec.py
    power = (power - power.min()) / (power.max() + 1e-30)
    power = np.log10(power + 5e-2)

    # x-axis: each sample is SAMPLING_RATE seconds
    # express as fraction of day so tms_num = date2num(t_start) + tms_day
    n       = lfr_values.shape[0]
    tms_day = np.arange(n) * SAMPLING_RATE / 86400.0   # seconds → fraction of day

    return tms_day, freq, power


def _parse_date(stem: str):
    """Extract year and DOY from filename, e.g. lx_raw_ROGER_20220407_entln_15s → 2022, 97."""
    match = re.search(r"(\d{8})", stem)
    if match:
        dt = pd.to_datetime(match.group(1), format="%Y%m%d")
        return dt.year, dt.day_of_year
    return None, None


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    # apply spatial bounding box only — hour filter is applied at plot time
    df = df[
        (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX) &
        (df["latitude"]  >= LAT_MIN) & (df["latitude"]  <= LAT_MAX)
    ]
    return df.reset_index(drop=True)


def _lfr_15s(df: pd.DataFrame, day_start: pd.Timestamp, hmin=HOUR_MIN, hmax=HOUR_MAX) -> pd.Series:
    """LFR at native 15s resolution, covering only the hmin–hmax window."""
    t_start = day_start + pd.Timedelta(hours=hmin)
    t_end   = day_start + pd.Timedelta(hours=hmax) - pd.Timedelta(seconds=SAMPLING_RATE)
    full_idx = pd.date_range(start=t_start, end=t_end, freq=f"{SAMPLING_RATE}s")
    lfr = (
        df.groupby("time_stamp")
          .size()
          .rename("lfr")
          / SAMPLING_RATE
    )
    return lfr.reindex(full_idx, fill_value=0.0)


def _filter_region(df, lon0, lon1, lat0, lat1):
    mask = (
        (df["longitude"] >= lon0) & (df["longitude"] <= lon1) &
        (df["latitude"]  >= lat0) & (df["latitude"]  <= lat1)
    )
    return df[mask]


def _filter_hours(df, hmin, hmax):
    """Keep only rows within [hmin, hmax)."""
    hour = df["time_stamp"].dt.hour + df["time_stamp"].dt.minute / 60
    return df[(hour >= hmin) & (hour < hmax)]


# ── plot helpers ──────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors="black", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_ACCENT)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")


def _colored_lfr_line(ax, lfr: pd.Series, norm: Normalize, cmap):
    """Draw LFR line coloured by hour of day using a LineCollection."""
    times  = mdates.date2num(lfr.index.to_pydatetime())
    values = lfr.values
    hours  = lfr.index.hour + lfr.index.minute / 60

    points   = np.array([times, values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.1)
    lc.set_array(hours[:-1])
    ax.add_collection(lc)

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, max(values.max() * 1.1, 0.01))
    return lc


# ── main explorer class ───────────────────────────────────────────────────────
class LFRExplorer:

    def __init__(self, files: list):
        self.files  = files
        self.idx    = 0
        self.region = None
        self._df    = None
        self._norm  = Normalize(vmin=0, vmax=24)
        self._cmap  = matplotlib.colormaps[CMAP]
        self._hmin  = HOUR_MIN
        self._hmax  = HOUR_MAX

        self._build_figure()
        self._load_day(0)
        plt.show()

    # ── figure layout ─────────────────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(BG_DARK)

        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[0.06, 1],
            hspace=0.08,
        )

        # ── button bar ────────────────────────────────────────────────────────
        btn_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[0], wspace=0.05,
        )
        ax_prev  = self.fig.add_subplot(btn_gs[0])
        ax_title = self.fig.add_subplot(btn_gs[1])
        ax_next  = self.fig.add_subplot(btn_gs[2])

        for ax in (ax_prev, ax_title, ax_next):
            ax.set_facecolor(BG_DARK)
            ax.axis("off")

        self.ax_title = ax_title
        self.btn_prev = Button(ax_prev, "◀  Prev", color="#DDDDDD", hovercolor="#BBBBBB")
        self.btn_next = Button(ax_next, "Next  ▶", color="#DDDDDD", hovercolor="#BBBBBB")
        self.btn_prev.label.set_color("black")
        self.btn_next.label.set_color("black")
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

        # ── content grid ──────────────────────────────────────────────────────
        content_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[1],
            width_ratios=[1, 1.3], wspace=0.28,
        )

        # left: scatter + colorbar
        left_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=content_gs[0],
            height_ratios=[1, 0.055], hspace=0.08,
        )
        self.ax_map  = self.fig.add_subplot(left_gs[0])
        self.ax_cbar = self.fig.add_subplot(left_gs[1])
        self.ax_map.set_aspect("equal", adjustable="box")

        # right: LFR (top) + wavelet (bottom)
        right_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=content_gs[1],
            height_ratios=[1, 1], hspace=0.42,
        )
        self.ax_lfr  = self.fig.add_subplot(right_gs[0])
        self.ax_spec = self.fig.add_subplot(right_gs[1])

        # wavelet placeholder
        _style_ax(self.ax_spec)
        self.ax_spec.text(
            0.5, 0.5, "Select a region to compute wavelet",
            transform=self.ax_spec.transAxes,
            ha="center", va="center",
            color="#555577", fontsize=9, style="italic",
        )
        self.ax_spec.set_xticks([])
        self.ax_spec.set_yticks([])

        # ── reset button ──────────────────────────────────────────────────────
        ax_reset = self.fig.add_axes([0.88, 0.01, 0.10, 0.030])
        self.btn_reset = Button(ax_reset, "Reset region", color="#DDDDDD", hovercolor="#BBBBBB")
        self.btn_reset.label.set_color("black")
        self.btn_reset.label.set_fontsize(8)
        self.btn_reset.on_clicked(self._on_reset)

        # ── rectangle selector ────────────────────────────────────────────────
        self.selector = RectangleSelector(
            self.ax_map, self._on_select,
            useblit=True, button=[1],
            minspanx=0.01, minspany=0.01,
            spancoords="data", interactive=True,
            props=dict(facecolor=C_REGION, edgecolor=C_REGION, alpha=0.2, fill=True),
        )

        # ── hour colorbar ─────────────────────────────────────────────────────
        cb = self.fig.colorbar(
            ScalarMappable(norm=self._norm, cmap=self._cmap),
            cax=self.ax_cbar, orientation="horizontal",
        )
        cb.set_label("Hour of day (UTC)", color="black", fontsize=8)
        cb.set_ticks(range(0, 25, 3))
        cb.ax.tick_params(colors="black", labelsize=7)
        cb.outline.set_edgecolor(C_ACCENT)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── data loading ──────────────────────────────────────────────────────────
    def _load_day(self, idx: int):
        self.idx    = idx % len(self.files)
        self._df    = _load(self.files[self.idx])
        self.region = None
        self._redraw()

    # ── drawing ───────────────────────────────────────────────────────────────
    def _redraw(self):
        self._draw_map()
        self._draw_lfr()
        self._draw_wavelet()
        self._update_title()
        self.fig.canvas.draw_idle()

    def _draw_map(self):
        self.ax_map.cla()
        self.ax_map.set_facecolor(BG_PANEL)
        self.ax_map.set_aspect("equal", adjustable="box")

        df       = _filter_hours(self._df, self._hmin, self._hmax)
        hour_dec = df["time_stamp"].dt.hour + df["time_stamp"].dt.minute / 60

        self.ax_map.scatter(
            df["longitude"], df["latitude"],
            c=hour_dec, cmap=CMAP, norm=self._norm,
            s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
            linewidths=0, rasterized=True,
        )

        if self.region is not None:
            lon0, lon1, lat0, lat1 = self.region
            rect = Rectangle(
                (lon0, lat0), lon1 - lon0, lat1 - lat0,
                linewidth=1.5, edgecolor=C_REGION,
                facecolor=C_REGION, alpha=0.15,
            )
            self.ax_map.add_patch(rect)
            n = len(_filter_region(df, lon0, lon1, lat0, lat1))
            self.ax_map.set_title(
                f"lon [{lon0:.2f}, {lon1:.2f}]  lat [{lat0:.2f}, {lat1:.2f}]"
                f"  —  {n:,} flashes",
                color="black", fontsize=9,
            )
        else:
            self.ax_map.set_title(
                f"{len(df):,} flashes  —  draw a rectangle to select a region",
                color="black", fontsize=9,
            )

        self.ax_map.set_xlabel("Longitude", color="black")
        self.ax_map.set_ylabel("Latitude",  color="black")
        self.ax_map.tick_params(colors="black", labelsize=8)
        for spine in self.ax_map.spines.values():
            spine.set_edgecolor(C_ACCENT)

        self.selector.ax = self.ax_map

    def _draw_lfr(self):
        self.ax_lfr.cla()
        _style_ax(self.ax_lfr)

        df = _filter_hours(self._df, self._hmin, self._hmax)
        if self.region is not None:
            lon0, lon1, lat0, lat1 = self.region
            df = _filter_region(df, lon0, lon1, lat0, lat1)

        if df.empty:
            self.ax_lfr.text(
                0.5, 0.5, "No flashes in selected region",
                transform=self.ax_lfr.transAxes,
                ha="center", va="center", color="black", fontsize=10,
            )
            return

        day_start = self._df["time_stamp"].dt.normalize().iloc[0]
        t_start   = day_start + pd.Timedelta(hours=self._hmin)
        t_end     = day_start + pd.Timedelta(hours=self._hmax)
        lfr       = _lfr_15s(df, day_start, self._hmin, self._hmax)

        _colored_lfr_line(self.ax_lfr, lfr, self._norm, self._cmap)

        self.ax_lfr.set_xlim(
            mdates.date2num(t_start.to_pydatetime()),
            mdates.date2num(t_end.to_pydatetime()),
        )
        self.ax_lfr.set_ylim(bottom=0)

        label = "Region LFR" if self.region is not None else f"Full domain LFR ({SAMPLING_RATE}s)"
        self.ax_lfr.set_title(label, color="black", fontsize=9)
        self.ax_lfr.set_ylabel("LFR  (flashes / s)", color="black")
        self.ax_lfr.set_xlabel("Hour (UTC)", color="black")

        self.ax_lfr.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        self.ax_lfr.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
        self.ax_lfr.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        self.ax_lfr.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.3, color="gray")
        self.ax_lfr.grid(axis="x", linestyle=":",  linewidth=0.3, alpha=0.25, color="gray")

    def _draw_wavelet(self):
        self.ax_spec.cla()
        _style_ax(self.ax_spec)

        df = _filter_hours(self._df, self._hmin, self._hmax)
        if self.region is not None:
            lon0, lon1, lat0, lat1 = self.region
            df = _filter_region(df, lon0, lon1, lat0, lat1)

        if df.empty:
            self.ax_spec.text(
                0.5, 0.5, "No flashes in selected region",
                transform=self.ax_spec.transAxes,
                ha="center", va="center", color="black", fontsize=9,
            )
            self.ax_spec.set_xticks([])
            self.ax_spec.set_yticks([])
            return

        day_start = self._df["time_stamp"].dt.normalize().iloc[0]
        lfr_vals  = _lfr_15s(df, day_start, self._hmin, self._hmax).values   # native 15s, same as f2_spec.py

        result = _compute_wavelet(lfr_vals)

        if result is None:
            self.ax_spec.text(
                0.5, 0.5, "Insufficient data for wavelet",
                transform=self.ax_spec.transAxes,
                ha="center", va="center", color="black", fontsize=9,
            )
            self.ax_spec.set_xticks([])
            self.ax_spec.set_yticks([])
            return

        tms_day, freq, power = result

        # convert fraction-of-day to matplotlib date numbers for alignment with LFR
        day_start = self._df["time_stamp"].dt.normalize().iloc[0]
        t_start   = day_start + pd.Timedelta(hours=self._hmin)
        t_end     = day_start + pd.Timedelta(hours=self._hmax)
        tms_num   = (mdates.date2num(t_start.to_pydatetime())
                     + tms_day)   # tms_day in [0,1] = fraction of window

        # contourf — identical to f2_spec.py
        cf = self.ax_spec.contourf(
            tms_num, freq, power,
            cmap="nipy_spectral", levels=20,
        )

        # x-axis aligned with LFR panel
        self.ax_spec.set_xlim(
            mdates.date2num(t_start.to_pydatetime()),
            mdates.date2num(t_end.to_pydatetime()),
        )
        self.ax_spec.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        self.ax_spec.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
        self.ax_spec.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

        self.ax_spec.set_xlabel("Hour (UTC)", color="black")
        self.ax_spec.set_ylabel("Frequency (mHz)", color="black")
        self.ax_spec.set_title(
            "Wavelet power — LFR  (Morlet, 2–10 mHz)",
            color="black", fontsize=9,
        )
        self.ax_spec.grid(axis="x", linestyle=":", linewidth=0.3, alpha=0.25, color="gray")

    def _update_title(self):
        self.ax_title.cla()
        self.ax_title.axis("off")
        stem  = self.files[self.idx].stem
        year, doy = _parse_date(stem)
        n     = self.idx + 1
        total = len(self.files)
        date_str = f"Year {year}  —  DOY {doy}" if year else stem
        self.ax_title.text(
            0.5, 0.5,
            f"{date_str}   [{n} / {total}]",
            transform=self.ax_title.transAxes,
            ha="center", va="center",
            color="black", fontsize=11, fontweight="bold",
        )

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _on_select(self, eclick, erelease):
        lon0 = min(eclick.xdata, erelease.xdata)
        lon1 = max(eclick.xdata, erelease.xdata)
        lat0 = min(eclick.ydata, erelease.ydata)
        lat1 = max(eclick.ydata, erelease.ydata)
        self.region = (lon0, lon1, lat0, lat1)
        self._draw_map()
        self._draw_lfr()
        self._draw_wavelet()
        self.fig.canvas.draw_idle()

    def _on_prev(self, _event):
        self._load_day(self.idx - 1)

    def _on_next(self, _event):
        self._load_day(self.idx + 1)

    def _on_reset(self, _event):
        self.region = None
        self._redraw()

    def _on_key(self, event):
        if event.key == "left":
            self._load_day(self.idx - 1)
        elif event.key == "right":
            self._load_day(self.idx + 1)
        elif event.key in ("r", "R"):
            self.region = None
            self._redraw()



# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parquet_dir = Path(PARQUET_DIR)
    files = sorted(parquet_dir.glob(f"*_{SAMPLING_RATE}s.parquet"))

    if not files:
        print(f"No *_{SAMPLING_RATE}s.parquet files found in: {parquet_dir}")
    else:
        print(f"Found {len(files)} file(s). Launching explorer...")
        explorer = LFRExplorer(files)