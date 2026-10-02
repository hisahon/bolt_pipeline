"""
lightning_keogram.py

Keograma de raios: soma TODOS os raios de cada instante de tempo em lat/lon.

  - Painel de cima  : LATITUDE  x tempo (soma sobre todas as longitudes)
  - Painel de baixo : LONGITUDE x tempo (soma sobre todas as latitudes)

Eixo x em horas UTC. A resolucao temporal por padrao e a do proprio parquet
(NEW_SAMPLING_RATE, em s).

Saida: data/keogram/{YYYYMMDD}/keograma_hex_{YYYYMMDD}_{HEX_TIME_MIN}min_{HEX_DEG}deg.png
Mapeamento em HEXBIN: cor = contagem de raios por hexagono (escala sqrt, log ou linear, ver COLOR_SCALE).

Basta ajustar o CONFIG e rodar.
"""

import matplotlib
matplotlib.use("Agg")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_DIR       = Path(r"data\parquet")
OUT_DIR           = Path(r"data\keogram")
NEW_SAMPLING_RATE = 15

TARGET_DATE = "20211023"        # 'YYYYMMDD'
TIME_RANGE  = None              # (tbeg, tend) em horas UTC decimais, ex.: (15, 18.5); None = dia todo
TIME_RANGE  = (15,18)              # (tbeg, tend) em horas UTC decimais, ex.: (15, 18.5); None = dia todo

# ROI (graus). Se NAO for definido (linha comentada/removida) ou for None, o valor
# e extraido dos proprios dados de raios: min/max de lat/lon (na janela TIME_RANGE,
# se houver), arredondado para fora na resolucao do keograma. Cada limite e
# independente: voce pode fixar so alguns e deixar o resto automatico.
ROI_LAT_MIN, ROI_LAT_MAX = -30, -20
ROI_LON_MIN, ROI_LON_MAX = -60, -50

# Janela do LFR total usada so no resumo impresso no console (pico de LFR); nao e plotada.
TIME_BIN_SEC = NEW_SAMPLING_RATE

# Keograma em HEXBIN (contagem de raios por hexagono; escala de cor sqrt, log ou linear).
# Tamanho do hexagono: largura em minutos (eixo do tempo) e altura em graus (lat/lon).
# Hexagonos maiores = imagem mais suave; menores = mais detalhe (e mais ruido).
HEX_TIME_MIN = 2.
HEX_DEG      = 1.
# Resolucao usada so para arredondar o ROI extraido dos dados (graus)
LAT_BIN_DEG = 0.1
LON_BIN_DEG = 0.1

CMAP = "viridis"
# Escala de cor da contagem por hexagono: "sqrt" (raiz quadrada), "log" ou "linear"
COLOR_SCALE = "sqrt"
# Limite superior fixo da cor (contagens/hexagono). None = maximo dos dados.
VMAX = None

MARK_TIMES       = []           # horarios a destacar com linha vertical, ex.: ["16:30"] (UTC)

FONT_SIZE = 13
# ─────────────────────────────────────────────────────────────────────────────

mpl.rcParams["font.size"] = FONT_SIZE


# ── utilidades ────────────────────────────────────────────────────────────────
def parse_hms(s: str) -> float:
    parts = [int(x) for x in s.split(":")]
    h, m, sec = (parts + [0, 0])[:3]
    return h + m / 60.0 + sec / 3600.0


def _fmt_hours(ax):
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda h, _: f"{int(h) % 24:02d}:{int(round((h % 1) * 60)):02d}")
    )


def _time_ticks(lo: float, hi: float) -> np.ndarray:
    """Ticks de horario 'redondos' (5/10/15/30 min, 1/2/3/6 h) em ~<=12 marcas."""
    span = hi - lo
    step = next((s for s in (1/12, 1/6, 1/4, 1/2, 1, 2, 3, 6) if span / s <= 12), 6)
    first = np.ceil(lo / step) * step
    return np.arange(first, hi + 1e-9, step)


def _savefig(fig, out_path: Path):
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


# ── leitura ───────────────────────────────────────────────────────────────────
def resolve_roi(df: pd.DataFrame) -> None:
    """
    Preenche ROI_LAT_MIN/MAX e ROI_LON_MIN/MAX que estiverem ausentes (nao definidos)
    ou None, a partir dos dados de raios. Os limites ja definidos sao respeitados
    e usados para restringir a extracao dos demais.
    """
    names = ("ROI_LAT_MIN", "ROI_LAT_MAX", "ROI_LON_MIN", "ROI_LON_MAX")
    cur = {n: globals().get(n) for n in names}
    if all(v is not None for v in cur.values()):
        return

    d = df
    if TIME_RANGE:                                    # so a janela de tempo que sera plotada
        h0, h1 = TIME_RANGE
        d = d[(d["t_hours"] >= h0) & (d["t_hours"] < h1)]
    # respeita limites ja fixados ao extrair os que faltam
    if cur["ROI_LAT_MIN"] is not None: d = d[d["latitude"] >= cur["ROI_LAT_MIN"]]
    if cur["ROI_LAT_MAX"] is not None: d = d[d["latitude"] <= cur["ROI_LAT_MAX"]]
    if cur["ROI_LON_MIN"] is not None: d = d[d["longitude"] >= cur["ROI_LON_MIN"]]
    if cur["ROI_LON_MAX"] is not None: d = d[d["longitude"] <= cur["ROI_LON_MAX"]]
    lat, lon = d["latitude"].dropna(), d["longitude"].dropna()
    if lat.empty or lon.empty:
        raise RuntimeError("Sem raios para extrair o ROI (confira TIME_RANGE / limites do ROI)")

    auto = {
        "ROI_LAT_MIN": np.floor(lat.min() / LAT_BIN_DEG) * LAT_BIN_DEG,
        "ROI_LAT_MAX": np.ceil(lat.max() / LAT_BIN_DEG) * LAT_BIN_DEG,
        "ROI_LON_MIN": np.floor(lon.min() / LON_BIN_DEG) * LON_BIN_DEG,
        "ROI_LON_MAX": np.ceil(lon.max() / LON_BIN_DEG) * LON_BIN_DEG,
    }
    for n in names:
        globals()[n] = cur[n] if cur[n] is not None else float(np.round(auto[n], 6))
    src = ", ".join(n for n in names if cur[n] is None)
    print(f"ROI extraido dos dados ({src}): "
          f"lat [{ROI_LAT_MIN:g}, {ROI_LAT_MAX:g}]  lon [{ROI_LON_MIN:g}, {ROI_LON_MAX:g}]")


def load_lightning(date_str: str, day: datetime) -> pd.DataFrame:
    files = sorted(PARQUET_DIR.glob(f"*{date_str}*_{NEW_SAMPLING_RATE}s.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet de raios para {date_str} em {PARQUET_DIR}")
    df = pd.read_parquet(files[0], columns=["time_stamp", "latitude", "longitude"])
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    if getattr(df["time_stamp"].dt, "tz", None) is not None:
        df["time_stamp"] = df["time_stamp"].dt.tz_convert(None)
    df["t_hours"] = (df["time_stamp"] - pd.Timestamp(day)).dt.total_seconds() / 3600.0
    resolve_roi(df)
    df = df[
        df["latitude"].between(ROI_LAT_MIN, ROI_LAT_MAX)
        & df["longitude"].between(ROI_LON_MIN, ROI_LON_MAX)
    ].copy()
    print(f"Arquivo: {files[0].name} | {len(df):,} raios dentro da ROI")
    return df


# ── keograma ──────────────────────────────────────────────────────────────────
def plot_keogram(df: pd.DataFrame, date_str: str, out_path: Path):
    h0, h1 = TIME_RANGE if TIME_RANGE else (0.0, 24.0)
    d = df[(df["t_hours"] >= h0) & (df["t_hours"] < h1)]
    if d.empty:
        raise RuntimeError("Nenhum raio na janela/ROI escolhidos (confira TIME_RANGE / ROI)")

    # total de raios no tempo (so para o resumo do console), na resolucao do parquet
    tb_h = TIME_BIN_SEC / 3600.0
    n_t = int(round((h1 - h0) / tb_h))
    t_edges = h0 + np.arange(n_t + 1) * tb_h
    total, _ = np.histogram(d["t_hours"], bins=t_edges)

    # nº de hexagonos em x (tempo) e y (lat/lon) a partir do tamanho desejado do hexagono
    nx = max(1, int(round((h1 - h0) * 60.0 / HEX_TIME_MIN)))
    ny_lat = max(1, int(round((ROI_LAT_MAX - ROI_LAT_MIN) / HEX_DEG)))
    ny_lon = max(1, int(round((ROI_LON_MAX - ROI_LON_MIN) / HEX_DEG)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

    hbs = []
    for ax, col, lo, hi, ny, ylab in (
        (axes[0], "latitude", ROI_LAT_MIN, ROI_LAT_MAX, ny_lat, "Latitude (°)"),
        (axes[1], "longitude", ROI_LON_MIN, ROI_LON_MAX, ny_lon, "Longitude (°)"),
    ):
        hb = ax.hexbin(d["t_hours"], d[col], gridsize=(nx, ny), extent=(h0, h1, lo, hi),
                       mincnt=1, cmap=CMAP, linewidths=0.0, rasterized=True)
        hbs.append(hb)
        ax.set_ylabel(ylab)
        ax.set_facecolor("white")

    # mesma escala de cor (COLOR_SCALE) nos dois paineis
    vmax = VMAX if VMAX is not None else max(float(hb.get_array().max()) for hb in hbs)
    vmax = max(vmax, 2.0)
    norms = {
        "sqrt": PowerNorm(gamma=0.5, vmin=1, vmax=vmax),
        "log": LogNorm(vmin=1, vmax=vmax),
        "linear": Normalize(vmin=1, vmax=vmax),
    }
    if COLOR_SCALE not in norms:
        raise ValueError(f'COLOR_SCALE invalido: {COLOR_SCALE!r} (use "sqrt", "log" ou "linear")')
    for ax, hb in zip(axes, hbs):
        hb.set_norm(norms[COLOR_SCALE])
        fig.colorbar(hb, ax=ax, pad=0.01, label=f"counts / hexagon ({HEX_TIME_MIN:g} min × {HEX_DEG:g}°)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(h0, h1)
        for t in MARK_TIMES:
            ax.axvline(parse_hms(t), color="black", lw=1.2, ls="--")
    axes[-1].set_xticks(_time_ticks(h0, h1))
    _fmt_hours(axes[-1])
    axes[-1].set_xlabel(f"Time (UTC) - {date_str[:4]}/{date_str[4:6]}/{date_str[6:]}")

    _savefig(fig, out_path)
    plt.close(fig)

    # resumo rapido
    k = int(np.argmax(total))
    print(f"Pico de LFR: {total[k] * 60.0 / TIME_BIN_SEC:,.0f} raios/min em "
          f"{int(t_edges[k]):02d}:{int((t_edges[k] % 1) * 60):02d}:{int(round(((t_edges[k] * 60) % 1) * 60)):02d} UTC")


def main():
    day = datetime.strptime(TARGET_DATE, "%Y%m%d")
    df = load_lightning(TARGET_DATE, day)
    out = OUT_DIR / TARGET_DATE / f"keograma_hex_{TARGET_DATE}_{HEX_TIME_MIN:g}min_{HEX_DEG:g}deg.png"
    plot_keogram(df, TARGET_DATE, out)


if __name__ == "__main__":
    main()
    print("\nAll done.")