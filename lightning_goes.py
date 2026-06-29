"""
plot_lightning_goes.py

Downloads GOES-16/19 channel-13 .nc files from CPTEC, reads them on the fly,
plots the IR brightness temperature with lightning strikes overlaid, then
deletes the .nc file. No .npz conversion.

One PNG per GOES frame saved in data/images/{YYYYMMDD}/

Folder layout:
    data/
        parquet/    ← resampled lightning .parquet files
        goes_tmp/   ← temporary .nc during download (auto-deleted after plot)
        images/     ← output PNG maps

Just set the CONFIG block below and run from Spyder.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
mpl.rcParams["font.size"]       = 11
mpl.rcParams["axes.titlesize"]  = 12
mpl.rcParams["axes.labelsize"]  = 10
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9

import os
import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin
# import netCDF4 as nc4


# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_DIR       = r"data\parquet"
GOES_TMP_DIR      = r"data\goes_tmp"
IMAGE_DIR         = r"data\images"
NEW_SAMPLING_RATE = 15
CHANNEL           = 13
GOES_TIMEOUT      = 30.0
CMAP_GOES         = "Blues_r"
VMIN_TEMP         = -80       # °C
VMAX_TEMP         = 30      # °C
# ─────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("OMP_NUM_THREADS", "1")

PARQUET_DIR  = Path(PARQUET_DIR)
GOES_TMP_DIR = Path(GOES_TMP_DIR)
IMAGE_DIR    = Path(IMAGE_DIR)


# ── satellite selection ───────────────────────────────────────────────────────
def _satellite(year: int, month: int) -> str:
    return "goes19" if year * 100 + month >= 202504 else "goes16"


def _month_url(year: int, month: int, channel: int) -> str:
    sat = _satellite(year, month)
    return (
        f"https://ftp.cptec.inpe.br/goes/{sat}"
        f"/retangular/ch{channel:02d}/{year:04d}/{month:02d}/"
    )


# ── HTTP session ──────────────────────────────────────────────────────────────
def _build_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )))
    s.headers.update({"User-Agent": "goes-lightning-plot/1.0"})
    return s


# ── list .nc files on server ──────────────────────────────────────────────────
def _list_nc_files(sess: requests.Session, url: str) -> list:
    try:
        r = sess.get(url, timeout=GOES_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return [
            a["href"] for a in soup.find_all("a", href=True)
            if a["href"].endswith(".nc") and not a["href"].startswith("?")
        ]
    except Exception as e:
        print(f"   [WARN] Could not list {url}: {e}")
        return []


# ── parse datetime from GOES filename ────────────────────────────────────────
def _dt_from_name(name: str) -> datetime | None:
    m = re.search(r"_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})", name)
    if m:
        y, mo, d, hh, mm = m.groups()
        return datetime(int(y), int(mo), int(d), int(hh), int(mm))
    return None


# ── download one .nc ─────────────────────────────────────────────────────────
def _download_nc(sess: requests.Session, url: str, nc_name: str) -> Path | None:
    GOES_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_nc   = GOES_TMP_DIR / nc_name
    tmp_part = tmp_nc.with_suffix(".nc.part")
    try:
        with sess.get(url, stream=True, timeout=GOES_TIMEOUT) as r:
            r.raise_for_status()
            with open(tmp_part, "wb") as f:
                for chunk in r.iter_content(8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_part, tmp_nc)
        return tmp_nc
    except Exception as e:
        print(f"   [ERR] Downloading {nc_name}: {e}")
        for p in (tmp_part, tmp_nc):
            try:
                if p.exists():
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        return None


# ── read .nc → arrays ────────────────────────────────────────────────────────
def _read_nc(nc_path: Path) -> tuple:
    import xarray as xr
    with xr.open_dataset(nc_path) as ds:
        lats = ds["lat"].values.astype("float32")
        lons = ds["lon"].values.astype("float32")
        temp = ds["Band1"].astype("float32").values / 100.0 - np.float32(273.13)
    return temp, lats, lons



# ── map plot ──────────────────────────────────────────────────────────────────
def plot_goes_lightning(goes_dt, temp, lat, lon, lightning, out_path):
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(
        temp,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=CMAP_GOES,
        vmin=VMIN_TEMP,
        vmax=VMAX_TEMP,
        aspect="auto",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Brightness temperature (°C)")

    if not lightning.empty:
        ax.scatter(
            lightning["longitude"], lightning["latitude"],
            s=1, color="red", alpha=0.6,
            linewidths=0, rasterized=True, zorder=3,
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    doy = goes_dt.timetuple().tm_yday
    ax.set_title(
        f"GOES CH{CHANNEL} IR  |  {goes_dt.strftime('%Y-%m-%d %H:%M')} UTC"
        f"  (DOY {doy:03d})  |  strikes: {len(lightning):,}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────
parquet_files = sorted(PARQUET_DIR.glob(f"*_{NEW_SAMPLING_RATE}s.parquet"))

if not parquet_files:
    print(f"No parquet files found in {PARQUET_DIR}")
else:
    sess = _build_session()

    for pq_path in parquet_files:
        m = re.search(r"(\d{8})", pq_path.name)
        if not m:
            print(f"Skipping {pq_path.name} — cannot parse date")
            continue

        file_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        print(f"\n{'='*60}")
        print(f"File : {pq_path.name}  ({file_date})")

        lightning = pd.read_parquet(pq_path)
        lightning["time_stamp"] = pd.to_datetime(lightning["time_stamp"])
        print(f"Lightning strikes: {len(lightning):,}")

        url   = _month_url(file_date.year, file_date.month, CHANNEL)
        names = _list_nc_files(sess, url)
        day_str    = f"_{file_date.year}{file_date.month:02d}{file_date.day:02d}"
        day_names  = [n for n in names if day_str in n]

        if not day_names:
            print(f"   [WARN] No GOES files found for {file_date}")
            continue

        # infer CPTEC sampling rate from the gaps between consecutive frames
        frame_dts = sorted([_dt_from_name(n) for n in day_names if _dt_from_name(n)])
        if len(frame_dts) >= 2:
            gaps = [(frame_dts[i+1] - frame_dts[i]).seconds // 60
                    for i in range(len(frame_dts) - 1)]
            cptec_sampling_min = min(gaps)
        else:
            cptec_sampling_min = 10   # fallback if only one frame
        print(f"   {len(day_names)} GOES frames found (sampling ~{cptec_sampling_min} min) — downloading and plotting...")

        for name in sorted(day_names):
            goes_dt = _dt_from_name(name)
            if not goes_dt:
                continue

            nc_path = _download_nc(sess, urljoin(url, name), name)
            if not nc_path:
                continue

            try:
                temp, lat, lon = _read_nc(nc_path)

                half_window = timedelta(minutes=cptec_sampling_min / 2)
                t_min  = pd.Timestamp(goes_dt - half_window)
                t_max  = pd.Timestamp(goes_dt + half_window)
                nearby = lightning[
                    (lightning["time_stamp"] >= t_min) &
                    (lightning["time_stamp"] <= t_max)
                ]

                doy      = goes_dt.timetuple().tm_yday
                out_name = (
                    f"{file_date.strftime('%Y%m%d')}_DOY{doy:03d}"
                    f"_{goes_dt.strftime('%H%M')}_CH{CHANNEL}.png"
                )
                out_path = IMAGE_DIR / file_date.strftime('%Y%m%d') / out_name

                plot_goes_lightning(goes_dt, temp, lat, lon, nearby, out_path)

            except Exception as e:
                print(f"   [ERR] Processing {name}: {e}")

            finally:
                try:
                    if nc_path.exists():
                        nc_path.unlink(missing_ok=True)
                except Exception:
                    pass

    print("\nAll done.")