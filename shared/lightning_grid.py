# -*- coding: utf-8 -*-
"""
lightning_grid.py
=================

Grades inteiras de contagem de raios + gráficos simples (SEM cartopy).

Fluxo
-----
1. ``read_day``       lê o parquet do dia (gerado por resampling_lightning.py).
2. ``extent_from_df`` extrai lat/lon mín/máx do arquivo.
3. ``build_grid``     conta ocorrências por (janela de tempo, lat, lon) numa
                      grade regular de ``res`` graus -> array inteiro 3D.
4. ``save_grid`` / ``load_grid``  .npz comprimido com grade + eixos lat/lon + tempos.
5. ``plot_*``         matplotlib puro (imshow), rápido.

Convenções
----------
* ``grid[k, i, j]`` = nº de ocorrências na janela ``k``, linha de latitude
  ``i``, coluna de longitude ``j``.
* ``lat`` e ``lon`` são os CENTROS das células, em ordem crescente
  (lat: sul -> norte), por isso os gráficos usam ``origin="lower"``.
* As bordas da grade ficam em múltiplos de ``res`` (ex.: -35.04, -35.00, ...),
  então grades de dias diferentes são alinhadas entre si (só o tamanho muda).
* ``times[k]`` = INÍCIO da janela ``k`` (datetime64[s]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

COLUMNS = ["latitude", "longitude", "time_stamp", "type_data"]

_MASK_CACHE: dict = {}


# =============================================================================
# Leitura
# =============================================================================
def find_day_file(parquet_dir, day) -> Optional[Path]:
    """Parquet do dia (busca recursiva por ``*AAAAMMDD*.parquet``)."""
    ymd = pd.Timestamp(day).strftime("%Y%m%d")
    files = sorted(Path(parquet_dir).rglob(f"*{ymd}*.parquet"))
    if not files:
        return None
    if len(files) > 1:
        print(f"[aviso] {len(files)} arquivos casam com {ymd}; usando {files[0].name}")
    return files[0]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    return df.dropna(subset=["latitude", "longitude", "time_stamp"])


def read_day(parquet_dir, day, use_types: Optional[Sequence[int]] = None,
             fix_spill: bool = True):
    """
    Ocorrências do dia ``day`` (00:00 <= t < 24:00). Retorna (df, caminho_do_arquivo).

    O resampling arredonda para o bin de 15 s mais próximo, então os últimos
    segundos de um dia podem sair com carimbo 00:00:00 do dia seguinte. Com
    ``fix_spill=True`` essas ocorrências saem do arquivo de D e entram no dia
    D+1 (lidas do arquivo de D, quando existe).
    """
    day = pd.Timestamp(day).normalize()
    nxt = day + pd.Timedelta(days=1)
    f = find_day_file(parquet_dir, day)
    if f is None:
        raise FileNotFoundError(f"Nenhum parquet para {day.date()} em {parquet_dir}")
    df = _read(f)
    df = df[df["time_stamp"] < nxt] if fix_spill else df

    if fix_spill:
        fp = find_day_file(parquet_dir, day - pd.Timedelta(days=1))
        if fp is not None:
            dp = _read(fp)
            dp = dp[dp["time_stamp"] >= day]
            if len(dp):
                df = pd.concat([df, dp], ignore_index=True)

    if use_types is not None:
        df = df[df["type_data"].isin(list(use_types))]
    return df.reset_index(drop=True), f


def extent_from_df(df: pd.DataFrame):
    """(lat_min, lat_max, lon_min, lon_max) dos dados."""
    return (float(df["latitude"].min()), float(df["latitude"].max()),
            float(df["longitude"].min()), float(df["longitude"].max()))


# =============================================================================
# Eixos e tamanho
# =============================================================================
def make_axes(lat_min, lat_max, lon_min, lon_max, res):
    """Centros das células (crescentes), com bordas alinhadas a múltiplos de ``res``."""
    lat0 = np.round(np.floor(lat_min / res + 1e-9) * res, 6)
    lon0 = np.round(np.floor(lon_min / res + 1e-9) * res, 6)
    nlat = int(np.floor((lat_max - lat0) / res + 1e-9)) + 1
    nlon = int(np.floor((lon_max - lon0) / res + 1e-9)) + 1
    lat = np.round(lat0 + res * (np.arange(nlat) + 0.5), 6)
    lon = np.round(lon0 + res * (np.arange(nlon) + 0.5), 6)
    return lat, lon


def grid_size_info(extent, res, window_min=10, dtype="uint16") -> dict:
    """Tamanho da grade (e memória em GB) ANTES de construí-la."""
    lat, lon = make_axes(*extent, res)
    n_win = int(np.ceil(1440 / window_min))
    gb = n_win * len(lat) * len(lon) * np.dtype(dtype).itemsize / 1e9
    return dict(n_win=n_win, nlat=len(lat), nlon=len(lon), cells=len(lat) * len(lon), gb=gb)


# =============================================================================
# Construção da grade
# =============================================================================
def build_grid(df: pd.DataFrame, day, res: float = 0.04, window_min: float = 10,
               extent=None, dtype: str = "uint16", max_gb: float = 3.0) -> dict:
    """
    Conta ocorrências por (janela, lat, lon).

    extent : (lat_min, lat_max, lon_min, lon_max) ou None = extraído do próprio ``df``.
    dtype  : tipo inteiro da grade (uint16 satura em 65535 por célula/janela;
             as saturações são contadas em ``n_saturated``).
    Retorna dict com: grid, lat, lon, times, res, window_min, day, extent,
    n_events, n_dropped, n_saturated.
    """
    day = pd.Timestamp(day).normalize()
    if extent is None:
        extent = extent_from_df(df)
    lat, lon = make_axes(*extent, res)
    nlat, nlon = len(lat), len(lon)
    n_win = int(np.ceil(1440 / window_min))

    dt = np.dtype(dtype)
    if dt.kind not in "iu":
        raise ValueError("dtype deve ser inteiro (ex.: 'uint16', 'int32').")
    gb = n_win * nlat * nlon * dt.itemsize / 1e9
    if gb > max_gb:
        raise MemoryError(
            f"A grade ocuparia {gb:.1f} GB ({n_win} janelas x {nlat} x {nlon}) > max_gb={max_gb}. "
            "Aumente a resolução (res), a janela (window_min) ou restrinja o extent.")

    lat_edge0 = lat[0] - res / 2
    lon_edge0 = lon[0] - res / 2
    iy = np.floor((df["latitude"].to_numpy(float) - lat_edge0) / res).astype(np.int64)
    ix = np.floor((df["longitude"].to_numpy(float) - lon_edge0) / res).astype(np.int64)
    sec = (df["time_stamp"] - day).dt.total_seconds().to_numpy()
    win = np.floor(sec / (window_min * 60.0)).astype(np.int64)

    ok = (iy >= 0) & (iy < nlat) & (ix >= 0) & (ix < nlon) & (win >= 0) & (win < n_win)
    n_dropped = int((~ok).sum())
    iy, ix, win = iy[ok], ix[ok], win[ok]

    grid = np.zeros((n_win, nlat, nlon), dtype=dt)
    cmax = np.iinfo(dt).max
    n_sat = 0

    order = np.argsort(win, kind="stable")
    flat = (iy * nlon + ix)[order]
    bounds = np.searchsorted(win[order], np.arange(n_win + 1))
    for w in range(n_win):
        a, b = bounds[w], bounds[w + 1]
        if a == b:
            continue
        cnt = np.bincount(flat[a:b], minlength=nlat * nlon)
        n_sat += int((cnt > cmax).sum())
        grid[w] = np.minimum(cnt, cmax).reshape(nlat, nlon)

    step = np.timedelta64(int(round(window_min * 60)), "s")
    times = day.to_datetime64().astype("datetime64[s]") + np.arange(n_win) * step

    return dict(grid=grid, lat=lat, lon=lon, times=times, res=float(res),
                window_min=float(window_min), day=str(day.date()),
                extent=np.asarray(extent, dtype=float), n_events=int(ok.sum()),
                n_dropped=n_dropped, n_saturated=n_sat)


# =============================================================================
# Salvar / carregar
# =============================================================================
def grid_filename(day, res, window_min) -> str:
    return f"lx_grid_{pd.Timestamp(day).strftime('%Y%m%d')}_{res:g}deg_{window_min:g}min.npz"


def save_grid(g: dict, path) -> Path:
    """Grade + eixos lat/lon + tempos num .npz comprimido."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **g)
    return path


def load_grid(path) -> dict:
    """Carrega o .npz (escalares voltam como int/float/str)."""
    out = {}
    with np.load(path) as z:
        for k in z.files:
            v = z[k]
            out[k] = v.item() if v.ndim == 0 else v
    return out


def make_day_grid(parquet_dir, day, out_dir, res: float = 0.04, window_min: float = 10,
                  extent=None, use_types=None, dtype: str = "uint16",
                  max_gb: float = 3.0, force: bool = False, verbose: bool = True) -> dict:
    """Constrói (ou reaproveita, se já existir) a grade de um dia e salva em ``out_dir``."""
    out = Path(out_dir) / grid_filename(day, res, window_min)
    if out.exists() and not force:
        if verbose:
            print(f"[cache] {out.name}")
        return load_grid(out)
    df, f = read_day(parquet_dir, day, use_types=use_types)
    g = build_grid(df, day, res=res, window_min=window_min, extent=extent,
                   dtype=dtype, max_gb=max_gb)
    save_grid(g, out)
    if verbose:
        print(f"[ok] {out.name} | {g['grid'].shape} {g['grid'].dtype} | "
              f"{g['n_events']:,} ocorrências | fora do extent: {g['n_dropped']}")
    return g


# =============================================================================
# Gráficos (matplotlib puro)
# =============================================================================
def _step(axis) -> float:
    return float(abs(axis[1] - axis[0])) if len(axis) > 1 else 1.0


def _overlay_mask(ax, mask_file, step: int = 10):
    """Contorno do domínio do ``mask_SouthAmerica.npz`` (referência geográfica sem cartopy)."""
    if mask_file is None or not Path(mask_file).exists():
        return
    key = (str(mask_file), step)
    if key not in _MASK_CACHE:
        dat = np.load(mask_file)
        la, lo, m = np.asarray(dat["lats"], float), np.asarray(dat["lons"], float), dat["mask"]
        la, lo, m = la[::step], lo[::step], m[::step, ::step].astype(np.uint8)
        if la[0] > la[-1]:
            la, m = la[::-1], m[::-1]
        if lo[0] > lo[-1]:
            lo, m = lo[::-1], m[:, ::-1]
        _MASK_CACHE[key] = (lo, la, m)
    lo, la, m = _MASK_CACHE[key]
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.contour(lo, la, m, levels=[0.5], colors="k", linewidths=0.6)
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def _colorbar(fig, ax, im, log):
    """Colorbar com rótulos inteiros simples (sem notação 10^n)."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="ocorrências / célula")
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    cb.ax.yaxis.set_major_formatter(fmt)
    if log:
        cb.ax.yaxis.set_minor_formatter(fmt)
    return cb


def plot_grid(counts, lat, lon, ax=None, title=None, log: bool = True, vmin: float = 1,
              vmax: Optional[float] = None, cmap: str = "turbo", mask_file=None,
              cbar: bool = True):
    """Mapa de uma grade 2D de contagens; células com 0 ficam em branco. Retorna (fig, ax, im)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    dy, dx = _step(lat), _step(lon)
    extent = [lon[0] - dx / 2, lon[-1] + dx / 2, lat[0] - dy / 2, lat[-1] + dy / 2]
    vmax = float(vmax) if vmax is not None else float(max(counts.max(), vmin + 1))
    vmax = max(vmax, vmin + 1)
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")

    im = ax.imshow(np.ma.masked_equal(counts, 0), origin="lower", extent=extent, cmap=cm,
                   norm=norm, interpolation="nearest",
                   aspect=1.0 / np.cos(np.deg2rad(float(np.mean(lat)))))
    _overlay_mask(ax, mask_file)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, linestyle=":", linewidth=0.4)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    if cbar:
        _colorbar(fig, ax, im, log)
    return fig, ax, im


def _save(fig, out_path, dpi):
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def window_title(g: dict, k: int) -> str:
    t = pd.Timestamp(g["times"][k])
    return f"{t:%H:%M %Y/%m/%d}  Raios = {int(g['grid'][k].sum())}"


def plot_window(g: dict, k: int, out_path=None, dpi: int = 120, **kw):
    """Uma janela de tempo (índice ``k``)."""
    fig, ax, im = plot_grid(g["grid"][k], g["lat"], g["lon"], title=window_title(g, k), **kw)
    _save(fig, out_path, dpi)
    return fig


def plot_sum(g: dict, k0: int = 0, k1: Optional[int] = None, out_path=None,
             dpi: int = 120, **kw):
    """Acumulado entre as janelas ``k0`` (inclusive) e ``k1`` (exclusive); default = dia todo."""
    k1 = g["grid"].shape[0] if k1 is None else k1
    tot = g["grid"][k0:k1].sum(axis=0, dtype=np.int64)
    t0, t1 = pd.Timestamp(g["times"][k0]), pd.Timestamp(g["times"][min(k1, len(g["times"])) - 1])
    title = f"{t0:%H:%M} a {t1:%H:%M} {t1:%Y/%m/%d}  Raios = {int(tot.sum())}"
    fig, ax, im = plot_grid(tot, g["lat"], g["lon"], title=title, **kw)
    _save(fig, out_path, dpi)
    return fig


def busiest_windows(g: dict, n: int = 6):
    """Índices das ``n`` janelas com mais ocorrências (em ordem cronológica) e a série de totais."""
    totals = g["grid"].reshape(g["grid"].shape[0], -1).sum(axis=1, dtype=np.int64)
    idx = np.sort(np.argsort(totals)[::-1][:n])
    return idx, totals


def plot_panels(g: dict, ks: Optional[Sequence[int]] = None, n: int = 6, ncols: int = 3,
                out_path=None, dpi: int = 110, cmap: str = "turbo", mask_file=None,
                log: bool = True):
    """Várias janelas lado a lado com a MESMA escala de cores (default: as ``n`` mais ativas)."""
    import matplotlib.pyplot as plt
    if ks is None:
        ks, _ = busiest_windows(g, n)
    ks = list(ks)
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.0 * nrows),
                             squeeze=False, constrained_layout=True)
    vmax = max(int(g["grid"][k].max()) for k in ks)
    im = None
    for ax, k in zip(axes.ravel(), ks):
        _, _, im = plot_grid(g["grid"][k], g["lat"], g["lon"], ax=ax, title=window_title(g, k),
                             vmax=vmax, cmap=cmap, mask_file=mask_file, cbar=False, log=log)
    for ax in axes.ravel()[len(ks):]:
        ax.axis("off")
    if im is not None:
        _colorbar(fig, axes.ravel().tolist(), im, log)
    _save(fig, out_path, dpi)
    return fig


def plot_timeseries(g: dict, out_path=None, dpi: int = 120):
    """Total de ocorrências por janela ao longo do dia."""
    import matplotlib.pyplot as plt
    _, totals = busiest_windows(g, 1)
    t = pd.to_datetime(g["times"])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(t, totals, where="post", linewidth=1.0)
    ax.set_ylabel(f"Ocorrências / {g['window_min']:g} min")
    ax.set_xlabel("Tempo (UTC)")
    ax.set_title(f"{g['day']}  Total = {int(totals.sum()):,}", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.4)
    fig.tight_layout()
    _save(fig, out_path, dpi)
    return fig
