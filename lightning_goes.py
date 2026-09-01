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
mpl.rcParams["font.size"]       = 14
mpl.rcParams["axes.titlesize"]  = 14
mpl.rcParams["axes.labelsize"]  = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14


# if FONT_ENGINE == "usetex":
mpl.rcParams["text.usetex"]  = True
mpl.rcParams["font.family"]  = "serif"


import os
import re
import inspect
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
VMAX_TEMP         = 0      # °C

# ── seleção de data ────────────────────────────────────────────────────────
# TARGET_DATE : processa apenas um dia específico. Formato 'YYYYMMDD' ou None.
# DATE_RANGE  : processa um intervalo (início, fim), ambos 'YYYYMMDD'. Ignorado
#               se TARGET_DATE estiver definido. Use None para desativar.
# Se os dois forem None, processa todos os parquet encontrados (comportamento original).
TARGET_DATE       = "20260722"                       # ex: "20250115"
DATE_RANGE        = None                       # ex: ("20250110", "20250120")

# TARGET_TIME : se definido (formato 'HHMM', ex: "1230"), gera apenas o
#               frame GOES mais próximo desse horário, para cada dia sendo
#               processado (em vez de todos os frames do dia). Use None
#               para desativar (comportamento original: todos os frames).
TARGET_TIME       = None                       # ex: "1230"

# ── mapa (cartopy) ─────────────────────────────────────────────────────────
SHOW_STATES       = True      # desenha fronteiras estaduais (admin_1)
SHOW_COUNTRIES    = True      # desenha fronteiras de países
SHOW_COASTLINE    = True      # desenha linha de costa
MAP_RESOLUTION    = "50m"     # '110m' (rápido/grosseiro), '50m' (equilíbrio), '10m' (detalhado/lento)
STATE_LINEWIDTH   = 0.6
STATE_COLOR       = "dimgray"
BORDER_LINEWIDTH  = 0.8
BORDER_COLOR      = "black"

# ── recorte espacial (ROI) ────────────────────────────────────────────────
# Limita a imagem GOES e os raios a uma sub-região, em vez do domínio inteiro
# baixado. Defina CROP_TO_ROI = False para voltar a usar a extensão completa.
CROP_TO_ROI       = True
ROI_LAT_MIN       = -40
ROI_LAT_MAX       = -10
ROI_LON_MIN       = -70
ROI_LON_MAX       = -40

# ── raios (aparência) ──────────────────────────────────────────────────────
# 'scatter' : um ponto por raio (rápido de entender, mas alpha satura e some
#             a diferença entre áreas com muitos e poucos raios).
# 'density' : agrupa os raios em uma grade hexagonal (hexbin) e usa COR para
#             indicar contagem por célula — melhor pra enxergar densidade,
#             e mais leve pra desenhar quando há muitos pontos (raw/µs).
STRIKE_DISPLAY_MODE = "density"    # "scatter" ou "density"

# usados apenas se STRIKE_DISPLAY_MODE == "scatter"
STRIKE_SIZE       = 4
STRIKE_ALPHA      = 0.15
STRIKE_COLOR      = "red"

# usados apenas se STRIKE_DISPLAY_MODE == "density"
# HEXBIN_GRIDSIZE:
#   "auto" → lê a resolução nativa (espaçamento de pixel) do próprio arquivo
#            GOES lido e usa HEXBIN_RESOLUTION_FACTOR para escalar o número
#            de células a partir dela.
#   int    → número fixo de hexágonos no eixo x (comportamento manual antigo).
HEXBIN_GRIDSIZE          = "auto"
HEXBIN_RESOLUTION_FACTOR = 0.25   # 1.0 = mesma resolução do GOES; 0.5 = metade
                                  # da resolução (células com o dobro do tamanho
                                  # do pixel do GOES); 2.0 = o dobro da resolução.
HEXBIN_CMAP       = "gnuplot"   # contraste bom sobre o CMAP_GOES (Blues_r)
HEXBIN_MINCOUNT   = 1         # células com menos raios que isso ficam transparentes
HEXBIN_ALPHA      = 0.85

# Escala de cor fixa (mesmos limites em todos os frames, pra poder comparar
# densidade entre imagens diferentes). Deixe None em qualquer um dos dois
# para que o matplotlib normalize automaticamente por frame (comportamento
# padrão, mas cada PNG fica em uma escala diferente).
HEXBIN_VMIN       = 0        # nº mínimo de raios por célula (cor mais fraca)
HEXBIN_VMAX       = 100       # nº máximo de raios por célula (cor mais forte);
                              # células acima disso saturam na cor de topo do cmap
# ─────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("OMP_NUM_THREADS", "1")

PARQUET_DIR  = Path(PARQUET_DIR)
GOES_TMP_DIR = Path(GOES_TMP_DIR)
IMAGE_DIR    = Path(IMAGE_DIR)


# ── proveniência automática (script / função / storage) ───────────────────────
def _latex_escape(s: str) -> str:
    """
    Escapa caracteres especiais do LaTeX em texto DINAMICO (nomes de arquivo,
    de funcao, caminhos). Com text.usetex=True, '_' inicia um subscrito e
    '\\' inicia um comando -- um nome como 'plot_goes_lightning' ou um
    caminho Windows 'data\\images\\' quebra a renderizacao sem isso
    (foi exatamente o erro "Undefined control sequence" que voce teve).
    """
    s = s.replace("\\", "/")  # separador de path do Windows -> forward slash
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for char, esc in replacements.items():
        s = s.replace(char, esc)
    return s


def add_provenance_footer(fig, storage_path, x=0.9, y=0.01, **kwargs):
    """
    Adiciona rodapé com Script / Function / Storage no rodapé da figura,
    descobrindo automaticamente (via inspect) o arquivo e a função que
    chamaram esta rotina — evita rodapés dessincronizados ao copiar/colar
    entre scripts.
    """
    caller = inspect.stack()[1]
    script_name   = _latex_escape(os.path.relpath(caller.filename))
    function_name = _latex_escape(caller.function)
    storage_text  = _latex_escape(str(storage_path))

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


# ── recorte espacial (ROI) ─────────────────────────────────────────────────
def _crop_to_roi(temp: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple:
    """
    Recorta os arrays de temperatura/lat/lon para a janela definida em
    ROI_LAT_MIN/MAX e ROI_LON_MIN/MAX. Assume grade regular (lat e lon 1D),
    como fornecida pela pasta 'retangular' da CPTEC.
    """
    if not CROP_TO_ROI:
        return temp, lat, lon

    lat_mask = (lat >= ROI_LAT_MIN) & (lat <= ROI_LAT_MAX)
    lon_mask = (lon >= ROI_LON_MIN) & (lon <= ROI_LON_MAX)

    if not lat_mask.any() or not lon_mask.any():
        print("   [WARN] ROI fora da área coberta pelo arquivo GOES — usando extensão completa")
        return temp, lat, lon

    temp_cropped = temp[np.ix_(lat_mask, lon_mask)]
    return temp_cropped, lat[lat_mask], lon[lon_mask]


# ── resolução nativa do GOES (a partir do próprio arquivo) ─────────────────
_last_printed_resolution = None  # evita spam de print a cada frame


def _infer_goes_resolution(lat: np.ndarray, lon: np.ndarray) -> tuple:
    """
    Calcula o espaçamento (em graus) entre pixels consecutivos de lat/lon,
    lido diretamente do arquivo GOES já aberto (não é um valor fixo/chutado).
    Usa a mediana das diferenças pra ser robusto a eventuais falhas pontuais
    na grade.
    """
    lat_res = float(np.median(np.abs(np.diff(lat)))) if len(lat) > 1 else None
    lon_res = float(np.median(np.abs(np.diff(lon)))) if len(lon) > 1 else None
    return lat_res, lon_res


def _hexbin_gridsize_from_resolution(lat, lon, lon_min, lon_max, lat_min, lat_max):
    """
    Traduz a resolução nativa do GOES (graus/pixel) em um gridsize (nx, ny)
    para o hexbin, de forma que cada célula de densidade corresponda
    aproximadamente a um pixel do satélite dentro do ROI plotado.
    """
    global _last_printed_resolution

    lat_res, lon_res = _infer_goes_resolution(lat, lon)
    if not lat_res or not lon_res:
        return 60  # fallback se a grade tiver 1 único ponto por eixo

    # fator < 1 → células maiores que o pixel do GOES (grade mais grossa)
    # fator > 1 → células menores que o pixel do GOES (grade mais fina)
    lat_res_scaled = lat_res / HEXBIN_RESOLUTION_FACTOR
    lon_res_scaled = lon_res / HEXBIN_RESOLUTION_FACTOR

    nx = max(1, round((lon_max - lon_min) / lon_res_scaled))
    ny = max(1, round((lat_max - lat_min) / lat_res_scaled))

    resolution_key = (round(lat_res, 5), round(lon_res, 5), HEXBIN_RESOLUTION_FACTOR)
    if resolution_key != _last_printed_resolution:
        print(f"   Resolução nativa GOES detectada: ~{lat_res:.4f}° lat × {lon_res:.4f}° lon "
              f"(fator {HEXBIN_RESOLUTION_FACTOR}) → hexbin gridsize=({nx}, {ny})")
        _last_printed_resolution = resolution_key

    return (nx, ny)



# ── map plot (cartopy) ──────────────────────────────────────────────────────
def plot_goes_lightning(goes_dt, temp, lat, lon, lightning, out_path):
    if CROP_TO_ROI:
        lon_min, lon_max = ROI_LON_MIN, ROI_LON_MAX
        lat_min, lat_max = ROI_LAT_MIN, ROI_LAT_MAX
    else:
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 9))

    ax          = fig.add_axes([0.08, 0.16, 0.68, 0.76], projection=proj)
    cax_temp    = fig.add_axes([0.79, 0.16, 0.02, 0.76])
    cax_strikes = fig.add_axes([0.08, 0.1, 0.68, 0.02])

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    im = ax.imshow(
        temp,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        transform=proj,
        cmap=CMAP_GOES,
        vmin=VMIN_TEMP,
        vmax=VMAX_TEMP,
        interpolation="nearest",
        zorder=1,
    )
    cbar = fig.colorbar(im, cax=cax_temp)
    cbar.set_label("Brightness Temperature (°C)")

    # ── feições geográficas ────────────────────────────────────────────────
    if SHOW_COASTLINE:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "coastline", MAP_RESOLUTION,
                facecolor="none",
            ),
            edgecolor=BORDER_COLOR, linewidth=BORDER_LINEWIDTH, zorder=2,
        )

    if SHOW_COUNTRIES:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "cultural", "admin_0_boundary_lines_land", MAP_RESOLUTION,
                facecolor="none",
            ),
            edgecolor=BORDER_COLOR, linewidth=BORDER_LINEWIDTH, zorder=2,
        )

    if SHOW_STATES:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "cultural", "admin_1_states_provinces_lines", MAP_RESOLUTION,
                facecolor="none",
            ),
            edgecolor=STATE_COLOR, linewidth=STATE_LINEWIDTH, zorder=2,
        )

    if not lightning.empty:
        if STRIKE_DISPLAY_MODE == "density":
            if isinstance(HEXBIN_GRIDSIZE, str) and HEXBIN_GRIDSIZE.lower() == "auto":
                gridsize = _hexbin_gridsize_from_resolution(
                    lat, lon, lon_min, lon_max, lat_min, lat_max
                )
            else:
                gridsize = HEXBIN_GRIDSIZE

            hb = ax.hexbin(
                lightning["longitude"], lightning["latitude"],
                gridsize=gridsize,
                extent=[lon_min, lon_max, lat_min, lat_max],
                cmap=HEXBIN_CMAP,
                mincnt=HEXBIN_MINCOUNT,
                alpha=HEXBIN_ALPHA,
                vmin=HEXBIN_VMIN,
                vmax=HEXBIN_VMAX,
                transform=proj,
                linewidths=0.1,
                zorder=3,
            )
            cbar2 = fig.colorbar(hb, cax_strikes, orientation='horizontal')
            cbar2.set_label("Lightning Strike Density (counts per bin)")
        else:
            ax.scatter(
                lightning["longitude"], lightning["latitude"],
                s=STRIKE_SIZE, color=STRIKE_COLOR, alpha=STRIKE_ALPHA,
                transform=proj,
                linewidths=0, rasterized=True, zorder=3,
            )

    gl = ax.gridlines(
        draw_labels=True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False

    doy = goes_dt.timetuple().tm_yday
    ax.set_title(
        rf"GOES Channel {CHANNEL} (10.3 $\mu$m) $-$ Brightness Temperature $T_B$ $-$ {goes_dt.strftime('%Y-%m-%d')}" + "\n" +
        rf"\textbf{{{goes_dt.strftime('%H:%M')} UTC}}, "
        rf"\textbf{{DOY {doy:03d}}}  $-$ "
        rf"$N_{{\mathrm{{strikes}}}}$ = {len(lightning):,}"
    )


    add_provenance_footer(fig, storage_path=str(IMAGE_DIR) + os.sep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {out_path.name}")


# ── seleção de data: filtro de parquet ────────────────────────────────────────
def _extract_date(p: Path):
    m = re.search(r"(\d{8})", p.name)
    return datetime.strptime(m.group(1), "%Y%m%d").date() if m else None


def _filter_by_date(files: list) -> list:
    if TARGET_DATE:
        target = datetime.strptime(TARGET_DATE, "%Y%m%d").date()
        filtered = [p for p in files if _extract_date(p) == target]
        print(f"Filtro de data ativo: {TARGET_DATE} → {len(filtered)} arquivo(s) encontrado(s)")
        return filtered

    if DATE_RANGE:
        start = datetime.strptime(DATE_RANGE[0], "%Y%m%d").date()
        end   = datetime.strptime(DATE_RANGE[1], "%Y%m%d").date()
        filtered = [p for p in files if _extract_date(p) and start <= _extract_date(p) <= end]
        print(f"Filtro de intervalo ativo: {DATE_RANGE[0]}–{DATE_RANGE[1]} → {len(filtered)} arquivo(s) encontrado(s)")
        return filtered

    return files


# ── main ──────────────────────────────────────────────────────────────────────
parquet_files = sorted(PARQUET_DIR.glob(f"*_{NEW_SAMPLING_RATE}s.parquet"))
parquet_files = _filter_by_date(parquet_files)

if not parquet_files:
    print(f"No parquet files found in {PARQUET_DIR} (verifique TARGET_DATE / DATE_RANGE)")
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

        # ── filtro de horário: mantém só o frame mais próximo de TARGET_TIME ──
        if TARGET_TIME:
            target_hh, target_mm = int(TARGET_TIME[:2]), int(TARGET_TIME[2:])
            target_dt = datetime.combine(file_date, datetime.min.time()).replace(
                hour=target_hh, minute=target_mm
            )
            named_dts = [(n, _dt_from_name(n)) for n in day_names]
            named_dts = [(n, d) for n, d in named_dts if d is not None]
            best_name, best_dt = min(
                named_dts, key=lambda nd: abs((nd[1] - target_dt).total_seconds())
            )
            day_names = [best_name]
            print(f"   Filtro de horário ativo: {TARGET_TIME} → frame mais próximo: "
                  f"{best_dt.strftime('%H:%M')} ({best_name})")

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
                temp, lat, lon = _crop_to_roi(temp, lat, lon)

                half_window = timedelta(minutes=cptec_sampling_min / 2)
                t_min  = pd.Timestamp(goes_dt - half_window)
                t_max  = pd.Timestamp(goes_dt + half_window)
                nearby = lightning[
                    (lightning["time_stamp"] >= t_min) &
                    (lightning["time_stamp"] <= t_max)
                ]

                if CROP_TO_ROI:
                    nearby = nearby[
                        (nearby["latitude"]  >= ROI_LAT_MIN) & (nearby["latitude"]  <= ROI_LAT_MAX) &
                        (nearby["longitude"] >= ROI_LON_MIN) & (nearby["longitude"] <= ROI_LON_MAX)
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