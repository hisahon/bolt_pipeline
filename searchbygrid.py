"""
lightning_lfr_regions.py

Divide a ROI em N x N regioes iguais (padrao 4 x 4 = 16), conta o LFR
(Lightning Flash Rate, raios/min) de cada regiao ao longo do tempo e plota
tudo em uma unica figura.

Entrada : data/parquet/*{YYYYMMDD}*_{NEW_SAMPLING_RATE}s.parquet
          (colunas: time_stamp, latitude, longitude)
Saida   : data/lfr/{YYYYMMDD}/mapa_regioes_{N}x{N}.png        (mapa com a numeracao das regioes)
          data/lfr/{YYYYMMDD}/LFR_regioes_{YYYYMMDD}_{N}x{N}.png  (LFR de cada regiao, painel N x N)

Basta ajustar o bloco CONFIG e rodar.
"""

import matplotlib
matplotlib.use("Agg")

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_DIR       = Path(r"data\parquet")
OUT_DIR           = Path(r"data\lfr")
NEW_SAMPLING_RATE = 15

TARGET_DATE = "20211023"        # 'YYYYMMDD'
# TIME_RANGE  = None              # (tbeg, tend) em horas UTC decimais, ex.: (15, 18.5); None = dia todo
TIME_RANGE  = (15,18)              # (tbeg, tend) em horas UTC decimais, ex.: (15, 18.5); None = dia todo

# ROI (a mesma do script de referencia)
ROI_LAT_MIN, ROI_LAT_MAX = -30, -20
ROI_LON_MIN, ROI_LON_MAX = -60, -50

# Grade de regioes: N_DIV x N_DIV  (use 3, 4 ou 5 -> 9, 16 ou 25 regioes iguais).
# O tamanho da grid aparece no titulo/rotulo de cada figura e no nome dos arquivos.
N_DIV = 2

# Janela de contagem do LFR = resolucao do proprio parquet (NEW_SAMPLING_RATE, em s).
# LFR (raios/min) = nº de raios na janela * 60 / BIN_SEC
BIN_SEC = NEW_SAMPLING_RATE

MAP_RESOLUTION = "50m"          # resolucao das feicoes do mapa de regioes ('110m', '50m', '10m')

USE_TEX = False                 # True se quiser o mesmo estilo LaTeX do script original
FONT_SIZE = 14
# ─────────────────────────────────────────────────────────────────────────────

mpl.rcParams["font.size"] = FONT_SIZE
if USE_TEX:
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"


# ── leitura ───────────────────────────────────────────────────────────────────
def load_lightning(date_str: str) -> pd.DataFrame:
    files = sorted(PARQUET_DIR.glob(f"*{date_str}*_{NEW_SAMPLING_RATE}s.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet de raios para {date_str} em {PARQUET_DIR}")
    df = pd.read_parquet(files[0], columns=["time_stamp", "latitude", "longitude"])
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    if getattr(df["time_stamp"].dt, "tz", None) is not None:
        df["time_stamp"] = df["time_stamp"].dt.tz_convert(None)
    print(f"Arquivo: {files[0].name} | {len(df):,} raios")
    return df


# ── regioes ───────────────────────────────────────────────────────────────────
def assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra pela ROI e atribui a cada raio o indice da regiao (1..N_DIV^2),
    numerada em ordem de leitura: do canto NW ao SE (linha a linha, norte primeiro).
    """
    df = df[
        df["latitude"].between(ROI_LAT_MIN, ROI_LAT_MAX)
        & df["longitude"].between(ROI_LON_MIN, ROI_LON_MAX)
    ].copy()

    dlat = (ROI_LAT_MAX - ROI_LAT_MIN) / N_DIV
    dlon = (ROI_LON_MAX - ROI_LON_MIN) / N_DIV

    col = np.floor((df["longitude"] - ROI_LON_MIN) / dlon).astype(int).clip(0, N_DIV - 1)
    row_from_south = np.floor((df["latitude"] - ROI_LAT_MIN) / dlat).astype(int).clip(0, N_DIV - 1)
    row = (N_DIV - 1) - row_from_south            # 0 = linha mais ao norte

    df["region"] = row * N_DIV + col + 1
    return df


def region_bounds(region: int) -> tuple:
    """(lat_min, lat_max, lon_min, lon_max) da regiao."""
    dlat = (ROI_LAT_MAX - ROI_LAT_MIN) / N_DIV
    dlon = (ROI_LON_MAX - ROI_LON_MIN) / N_DIV
    row, col = divmod(region - 1, N_DIV)
    lat_max = ROI_LAT_MAX - row * dlat
    lon_min = ROI_LON_MIN + col * dlon
    return lat_max - dlat, lat_max, lon_min, lon_min + dlon


# ── LFR ───────────────────────────────────────────────────────────────────────
def compute_lfr(df: pd.DataFrame, day: datetime) -> pd.DataFrame:
    """
    Retorna DataFrame (indice = horario UTC do inicio de cada janela, em horas
    decimais; colunas = regioes) com o LFR em raios/min.
    """
    h0, h1 = TIME_RANGE if TIME_RANGE else (0.0, 24.0)
    sec = (df["time_stamp"] - pd.Timestamp(day)).dt.total_seconds()
    sec0, sec1 = int(round(h0 * 3600)), int(round(h1 * 3600))
    df = df[(sec >= sec0) & (sec < sec1)].copy()

    # bins inteiros na resolucao do parquet (NEW_SAMPLING_RATE segundos)
    df["bin"] = np.floor(sec[df.index] / BIN_SEC).astype(int) - sec0 // BIN_SEC
    n_bins = (sec1 - sec0) // BIN_SEC

    counts = (
        df.groupby(["bin", "region"]).size().unstack("region", fill_value=0)
        .reindex(index=range(n_bins), fill_value=0)
        .reindex(columns=range(1, N_DIV**2 + 1), fill_value=0)
    )
    counts.index = (sec0 + counts.index.to_numpy() * BIN_SEC) / 3600.0
    counts.index.name = "t_hours_utc"
    return counts * (60.0 / BIN_SEC)     # raios / min


# ── plots ─────────────────────────────────────────────────────────────────────
def _savefig(fig, out_path: Path):
    """
    Salva a figura. Se o Windows recusar abrir o arquivo (OSError: arquivo preso
    por outro programa, OneDrive, antivirus, etc.), tenta de novo com um sufixo
    de horario no nome em vez de derrubar a execucao inteira.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    except OSError as e:
        alt = out_path.with_name(f"{out_path.stem}_{datetime.now():%H%M%S}{out_path.suffix}")
        print(f"   [WARN] Nao foi possivel gravar {out_path.name} ({e}); tentando {alt.name}")
        fig.savefig(alt, dpi=130, bbox_inches="tight")
        out_path = alt
    print(f"   Saved -> {out_path}")


def _fmt_hours(ax):
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda h, _: f"{int(h) % 24:02d}:{int(round((h % 1) * 60)):02d}")
    )


def plot_grid(lfr: pd.DataFrame, date_str: str, out_path: Path):
    ymax = max(1.0, float(lfr.values.max()) * 1.05)
    fig, axes = plt.subplots(N_DIV, N_DIV, figsize=(14, 11), sharex=True, sharey=True)

    for region in lfr.columns:
        row, col = divmod(region - 1, N_DIV)
        ax = axes[row, col]
        ax.step(lfr.index, lfr[region], where="post", color="firebrick", lw=1.2, zorder=3)
        ax.fill_between(lfr.index, lfr[region], step="post", color="firebrick", alpha=0.25, zorder=2)

        # numero grande centralizado no painel (por cima da curva, translucido)
        ax.text(0.5, 0.5, f"{region}", transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT_SIZE * 20 / N_DIV, fontweight="bold",
                color="black", alpha=0.35, zorder=5)

        ax.set_ylim(0, ymax)
        ax.set_xlim(lfr.index[0], lfr.index[-1])
        ax.grid(alpha=0.3, zorder=0)
        _fmt_hours(ax)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="upper"))
        ax.tick_params(labelsize=FONT_SIZE - 4)
        ax.label_outer()          # ticks so na borda externa da grade

    # ticks de horario "redondos" (5 min, 10 min, 15 min, 30 min, 1 h, ...), fora das bordas
    # (com os paineis colados, um tick na borda colidiria com o do painel vizinho)
    lo, hi = float(lfr.index[0]), float(lfr.index[-1])
    span = hi - lo
    step = next((s for s in (1/12, 1/6, 1/4, 1/2, 1, 2, 3, 6, 12)
                 if span / s <= max(3, 12 // N_DIV)), 12)
    ticks = [t for t in mpl.ticker.MultipleLocator(step).tick_values(lo, hi)
             if lo + 0.04 * span < t < hi - 0.04 * span]
    axes[-1, 0].set_xticks(ticks)

    # paineis colados, sem titulos nem suptitle: maximiza a area de dados
    fig.subplots_adjust(left=0.075, right=0.995, top=0.995, bottom=0.075, wspace=0.0, hspace=0.0)
    fig.supxlabel(f"Time (UTC) - {date_str[:4]}/{date_str[4:6]}/{date_str[6:]} - grid {N_DIV}x{N_DIV}",
                  y=0.005, fontsize=FONT_SIZE - 2)
    fig.supylabel(f"LFR (flashes/min, {BIN_SEC}-s bins)", x=0.005, fontsize=FONT_SIZE - 2)
    _savefig(fig, out_path)
    plt.close(fig)


def plot_region_map(out_path: Path):
    """Mapa da ROI com as 16 regioes e o numero (R01..R16) centrado em cada uma."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_axes([0.1, 0.08, 0.85, 0.84], projection=proj)
    ax.set_extent([ROI_LON_MIN, ROI_LON_MAX, ROI_LAT_MIN, ROI_LAT_MAX], crs=proj)

    ax.add_feature(cfeature.LAND.with_scale(MAP_RESOLUTION), facecolor="#f2efe9", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale(MAP_RESOLUTION), facecolor="#dbe9f4", zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", MAP_RESOLUTION, facecolor="none"),
        edgecolor="black", linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", MAP_RESOLUTION, facecolor="none"),
        edgecolor="black", linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", MAP_RESOLUTION, facecolor="none"),
        edgecolor="dimgray", linewidth=0.6, zorder=2)

    # linhas que dividem as regioes
    dlat = (ROI_LAT_MAX - ROI_LAT_MIN) / N_DIV
    dlon = (ROI_LON_MAX - ROI_LON_MIN) / N_DIV
    for i in range(N_DIV + 1):
        ax.plot([ROI_LON_MIN, ROI_LON_MAX], [ROI_LAT_MIN + i * dlat] * 2,
                color="firebrick", lw=1.5, transform=proj, zorder=3)
        ax.plot([ROI_LON_MIN + i * dlon] * 2, [ROI_LAT_MIN, ROI_LAT_MAX],
                color="firebrick", lw=1.5, transform=proj, zorder=3)

    # numero centrado em cada regiao
    for region in range(1, N_DIV**2 + 1):
        la0, la1, lo0, lo1 = region_bounds(region)
        ax.text((lo0 + lo1) / 2, (la0 + la1) / 2, f"R{region:02d}",
                ha="center", va="center", fontsize=FONT_SIZE + 4, fontweight="bold",
                color="firebrick", transform=proj, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3, alpha=0.4, zorder=1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.linspace(ROI_LON_MIN, ROI_LON_MAX, N_DIV + 1))
    gl.ylocator = mticker.FixedLocator(np.linspace(ROI_LAT_MIN, ROI_LAT_MAX, N_DIV + 1))

    ax.set_title(f"Regions - grid {N_DIV}x{N_DIV} ({N_DIV**2} regions)")
    _savefig(fig, out_path)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    day = datetime.strptime(TARGET_DATE, "%Y%m%d")
    df = assign_regions(load_lightning(TARGET_DATE))
    print(f"Raios dentro da ROI: {len(df):,}")

    lfr = compute_lfr(df, day)
    print("Total de raios por regiao:")
    print((lfr.sum() * BIN_SEC / 60.0).round().astype(int).to_string())

    out_dir = OUT_DIR / TARGET_DATE
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_tag = f"{N_DIV}x{N_DIV}"          # marca o tamanho da grid no nome de todos os arquivos

    plot_region_map(out_dir / f"mapa_regioes_{grid_tag}.png")
    plot_grid(lfr, TARGET_DATE, out_dir / f"LFR_regioes_{TARGET_DATE}_{grid_tag}.png")


if __name__ == "__main__":
    main()
    print("\nAll done.")