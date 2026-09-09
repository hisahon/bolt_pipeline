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
mpl.rcParams["font.size"]       = 16
mpl.rcParams["axes.titlesize"]  = 16
mpl.rcParams["axes.labelsize"]  = 16
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16


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
import matplotlib.ticker as mticker
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import duckdb
# import netCDF4 as nc4

# módulo compartilhado do projeto (mesmo usado em visualize_hilbert_temporary.py)
from shared.config import dtec_ipp_dir


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
TARGET_DATE       = "20260723"                       # ex: "20250115"
DATE_RANGE        = None                       # ex: ("20250110", "20250120")

# TARGET_TIME : se definido (formato 'HHMM', ex: "1230"), gera apenas o
#               frame GOES mais próximo desse horário, para cada dia sendo
#               processado (em vez de todos os frames do dia). Use None
#               para desativar (comportamento original: todos os frames).
TARGET_TIME       = None                     # ex: "1230"



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
ROI_LAT_MIN       = -35
ROI_LAT_MAX       = -15
ROI_LON_MIN       = -65
ROI_LON_MAX       = -45

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

# ── IPPs de GNSS (trajetória) ───────────────────────────────────────────────
# Sobrepõe no mapa apenas a TRAJETÓRIA espacial (lat/lon) dos IPPs lidos dos
# arquivos dtec_ipp, coloridos pelo horário (t_hours) -- não pela magnitude
# do dTEC. É a mesma leitura usada em visualize_hilbert_temporary.py, mas
# aqui só precisamos de posição + tempo, não do valor de dTEC.
PLOT_IPP_TRAJECTORY = True
IPP_LAT_RANGE   = (ROI_LAT_MIN, ROI_LAT_MAX)  # por padrão, reusa o mesmo ROI do GOES
IPP_LON_RANGE   = (ROI_LON_MIN, ROI_LON_MAX)
IPP_TIME_RANGE  = None      # (tbeg, tend) em horas decimais UTC, ou None p/ o dia inteiro
IPP_STATIONS    = None      # lista de estações (ex.: ["ppte", "puma"]) ou None p/ todas
IPP_PAIRS       = None      # lista "estacao · PRN" (ex.: ["ppte · G12"]) ou None p/ todas
IPP_CMAP        = "turbo"   # escala de cor para o horário (t_hours)
IPP_MARKER_SIZE = 2
IPP_ALPHA       = 1.

# Só a escala de tempo do IPP fica DENTRO do mapa (inset), pequena -- as
# outras duas (temperatura do GOES, densidade de raios) continuam como
# estavam, fora do mapa. [x0, y0, largura, altura] em fração dos eixos do
# mapa (0,0 = canto inferior esquerdo do mapa; 1,1 = canto superior direito).
IPP_CBAR_RECT = [0.62, 0.95, 0.36, 0.03]

# Marca, por cima da trajetória do dia inteiro, a posição do IPP bem no
# horário (epoch) exato de CADA frame do GOES -- um ponto de destaque que
# "anda" ao longo da trajetória conforme os frames avançam no tempo.
IPP_EPOCH_MARKER      = True    # liga/desliga o marcador de epoch
IPP_EPOCH_MAX_GAP_MIN = 10.0    # ignora o marcador se o IPP mais próximo estiver a mais
                                # de X min do horário do frame (None = sem limite/sempre marca)
IPP_EPOCH_MARKER_STYLE = "o"
IPP_EPOCH_MARKER_SIZE  = 50
IPP_EPOCH_COLOR        = "blue"
IPP_EPOCH_EDGE_COLOR   = "black"
IPP_EPOCH_EDGE_WIDTH   = 1.2
# ─────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("OMP_NUM_THREADS", "1")

PARQUET_DIR  = Path(PARQUET_DIR)
GOES_TMP_DIR = Path(GOES_TMP_DIR)
IMAGE_DIR    = Path(IMAGE_DIR)





# ── manual_runs (opcional) ───────────────────────────────────────────────────
# Mesmo padrão do visualize_hilbert_temporary.py: cada entrada define o dia
# (year/doy), a janela de tempo (tbeg/tend) e a lista de pares "estação · PRN"
# a usar na trajetória de IPPs. Quando 'manual_runs' tem 1+ entradas, ele
# assume PRIORIDADE sobre TARGET_DATE/DATE_RANGE/IPP_TIME_RANGE/IPP_PAIRS —
# cada entrada roda como um dia isolado, com sua própria janela/lista de
# pares para o overlay de IPPs. Deixe [] (lista vazia) para desativar e usar
# o comportamento padrão de cima (TARGET_DATE/DATE_RANGE + IPP_TIME_RANGE/IPP_PAIRS globais).

def obter_parametros_execucao(data_str):
    """
    Recebe uma data no formato 'YYYYMMDD', calcula o ano e o DOY,
    e retorna uma lista contendo o dicionário com os parâmetros correspondentes.
    """
    try:
        # Converte a string para extrair ano e doy
        data_obj = datetime.strptime(data_str, "%Y%m%d")
        year = data_obj.year
        doy = data_obj.timetuple().tm_yday
    except ValueError:
        print("Erro: A data deve estar no formato 'YYYYMMDD'.")
        return []

    # Base de dados em formato de dicionário para busca rápida
    manual_runs = {
        
        
        (2023, 347): {
            "tbeg": 15, 
            "tend": 16.5,
            "pairs": [
                "rcro · E03", "py01 · G10", "prur · G27", 
                "prgu · E13", "prcl · E13", "msnv · G27", 
                "msmn · E13", "msbl · G27", "guai · G27"
            ]
        },
        
    # (2023,347): {
    #     "tbeg": 7.5,
    #     "tend": 10,
    #     "pairs": [
    #                        # # "uyar · G11",
    #                        # "uyar · G06",
    #                        # # "unsa · G06",
                           
    #                        # "tilc · G06",
    #                        # "uybu · G06",
    #                        # "meco · G06",
    #                        # "sjsp · G15",
    #                        # "teru · G11",
                           
    #                        # "tava · G06",

    #                        # "sur1 · G06",
                           
                           
    #                        # "rcro · E36",
    #                        # "cata · G11",
    #                        # "rcro · G11",
    #                        # "tucu · G11",
    #                        # "rcro · E11",
                           
    #                        # "puma · G06",
    #                        # "puma · R21",
    #                        # "tava · G11",
    #                        # "ojoa · G11",
                           
                           
    #                        "jvgo · G06",
    #                        "jvch · G11",
    #                        "jbal · G11",
    #                        "chac · R21",
                           
                           
    #                        "caep · G11",
    #                        "caep · G06",
    #                        "reco · G06",
    #                        "puma · G11",
    #                        "jvgo · G11",
    #                        "yema · G06",
    #                        "caep · R21",
    #                        "tatu · G11",
    #                        "tatu · G06",
    #                        "caep · R22"
    #                    ]
    #        },
        (2024,333): {
            "tbeg": 14,
            "tend": 16,
            "pairs": [
                            "each · R24",
                            "ifsc · R24",
                            "mgin · R24",
                            "mgla · R24",
                            "fede · G29",
                            "eesc · R24",


                            "mgrp · R24",
                            "mspp · R01",
                            "nela · R24",
                            "poli · R24",
                            "puma · G29",
                            "reco · G29",
                            "rspf · G18",
                            "sbal · G29",
                            "scch · G18",
                            "scla · G33",
                            "spcl · R24"
                            ]
            },
        (2026, 204): {
            "tbeg": 16, 
            "tend": 17 + 40/60,
            "pairs": [
                "scaq · R18", "stha · R18", "stha · G29", 
                "itai · R18", "itai · G29", "prcl · G29", 
                "puma · G11", "fosa · R18", "rosa · G29"
            ]
        },
        (2026, 203): {
            "tbeg": 14, 
            "tend": 16,
            "pairs": [
                "fosa · G11", "msdr · R24", "ppte · E23", 
                "ppte · G12", "ppte · G13", "prma · G12", 
                "prma · E23", "puma · R24", "py01 · G11"
            ]
        },
        (2026,204):{
            "tbeg": 16,
            "tend": 17 + 40/60,
                        "pairs": [
                                # "caep · R18",
                                # "neia · R18",
                                # "itai · G11",
                                # "rscl · E34",
                                # "rscl · G11",
                                "scaq · R18",
                                # "scbl · R18",
                                "stha · R18",
                                
                                "stha · G29",
                                # "uyvi · G29",
                                "itai · R18",
                                "itai · G29",

                                "prcl · G29",
                                "puma · G11",
                                # "eesc · G29",
                                "fosa · R18",
                                # "guai · R18",
                                "rosa · G29",
                                # "py01 · R18",
                    ]
        },
        
        (2021, 296): {
            "tbeg": 15 + 30/60, 
            "tend": 16 + 50/60,
            "pairs": [
                "coru · G11", "ilha · G29", "mspm · G12", 
                "msaq · G25", "msbl · R13", "msdr · E02", 
                "msdr · G25", "mspm · G25", "mspm · R13"
            ]
        }
    }
    
    # Busca pela chave (year, doy)
    dados = manual_runs.get((year, doy))
    
    # Se encontrar, formata a saída exatamente como solicitado
    if dados:
        return [
            {
                "year": year, 
                "doy": doy, 
                "tbeg": dados["tbeg"], 
                "tend": dados["tend"],
                "pairs": dados["pairs"]
            }
        ]
    else:
        return []

manual_runs = obter_parametros_execucao(TARGET_DATE)



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


# ── IPP trajectory reader (dtec_ipp parquet) ──────────────────────────────────
IPP_TIME_COL = "horario"   # datetime
IPP_LAT_COL  = "lts"       # IPP latitude  (deg)
IPP_LON_COL  = "lns"       # IPP longitude (deg)
IPP_PRN_COL  = "prn"       # satellite id


def _ipp_station_from_path(path: Path) -> str:
    return path.name.split("_dtecIPP_")[0]


def load_ipp_trajectory(
    year: int, doy: int, lat_range: tuple, lon_range: tuple,
    time_range: tuple | None = None, stations: list | None = None,
    pairs: list | None = None,
) -> pd.DataFrame:
    """
    Lê todos os *_dtecIPP_{year}{doy}.parquet da pasta do dia e devolve só
    posição (lat/lon) + horário de cada IPP -- sem precisar da coluna de
    dTEC, já que aqui o interesse é a TRAJETÓRIA espacial colorida pelo
    horário, não a magnitude da perturbação.
    """
    ipp_folder = Path(dtec_ipp_dir(year, doy))
    files = sorted(ipp_folder.glob(f"*_dtecIPP_{year}{doy}.parquet"))
    empty = pd.DataFrame(columns=[IPP_LAT_COL, IPP_LON_COL, "t_hours", "pair", "station"])
    if not files:
        print(f"   [WARN] Nenhum arquivo dtecIPP em {ipp_folder}")
        return empty

    if stations is not None:
        wanted = set(stations)
        files = [f for f in files if _ipp_station_from_path(f) in wanted]
        if not files:
            print(f"   [WARN] Nenhuma estação de IPP_STATIONS encontrada em {ipp_folder}")
            return empty

    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    file_list = [str(f) for f in files]

    con = duckdb.connect()
    query = f"""
        SELECT
            "{IPP_TIME_COL}" AS {IPP_TIME_COL},
            "{IPP_LAT_COL}"  AS {IPP_LAT_COL},
            "{IPP_LON_COL}"  AS {IPP_LON_COL},
            "{IPP_PRN_COL}"  AS {IPP_PRN_COL},
            regexp_extract(filename, '([^/\\\\]+)_dtecIPP_', 1) AS station
        FROM read_parquet($files, filename = true)
        WHERE "{IPP_LAT_COL}" BETWEEN $lat_lo AND $lat_hi
          AND "{IPP_LON_COL}" BETWEEN $lon_lo AND $lon_hi
          AND "{IPP_LAT_COL}" IS NOT NULL
          AND "{IPP_LON_COL}" IS NOT NULL
    """
    out = con.execute(query, {
        "files": file_list,
        "lat_lo": lat_lo, "lat_hi": lat_hi,
        "lon_lo": lon_lo, "lon_hi": lon_hi,
    }).df()
    con.close()

    if out.empty:
        return empty

    t = pd.to_datetime(out[IPP_TIME_COL])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    out["t_hours"] = t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0
    out["pair"] = out["station"].astype(str) + " · " + out[IPP_PRN_COL].astype(str)

    if time_range is not None:
        h0, h1 = time_range
        out = out[out["t_hours"].between(h0, h1)]

    if pairs is not None:
        out = out[out["pair"].isin(pairs)]

    return out.reset_index(drop=True)


def _hours_to_hhmm(h: float) -> str:
    """Converte hora decimal (ex.: 14.5) para string 'HH:MM' (ex.: '14:30')."""
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        mm = 0
        hh += 1
    hh %= 24
    return f"{hh:02d}:{mm:02d}"


def _nearest_ipp_epoch(
    ipp_day: pd.DataFrame, goes_dt: datetime, max_gap_min: float | None = None,
) -> pd.DataFrame:
    """
    Para cada par (estação · PRN) em ipp_day, acha o ponto com horário mais
    próximo de goes_dt -- usado para destacar a posição "ao vivo" do IPP no
    instante exato da figura, por cima da trajetória (que mostra o dia
    inteiro / a janela toda). Se max_gap_min for definido, descarta pares
    cujo ponto mais próximo esteja mais longe do que isso no tempo (evita
    marcar uma posição "congelada" quando não há dado perto do horário).
    """
    if ipp_day.empty:
        return ipp_day

    target_t = goes_dt.hour + goes_dt.minute / 60.0 + goes_dt.second / 3600.0
    rows = []
    for _, group in ipp_day.groupby("pair"):
        idx = (group["t_hours"] - target_t).abs().idxmin()
        row = group.loc[idx]
        gap_min = abs(row["t_hours"] - target_t) * 60.0
        if max_gap_min is not None and gap_min > max_gap_min:
            continue
        rows.append(row)

    if not rows:
        return ipp_day.iloc[0:0]
    return pd.DataFrame(rows)


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
def plot_goes_lightning(goes_dt, temp, lat, lon, lightning, out_path, ipp=None, ipp_epoch=None):
    if CROP_TO_ROI:
        lon_min, lon_max = ROI_LON_MIN, ROI_LON_MAX
        lat_min, lat_max = ROI_LAT_MIN, ROI_LAT_MAX
    else:
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())

    has_ipp = ipp is not None and not ipp.empty
    has_epoch = ipp_epoch is not None and not ipp_epoch.empty

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

    if has_ipp:
        sc_ipp = ax.scatter(
            ipp[IPP_LON_COL], ipp[IPP_LAT_COL],
            c=ipp["t_hours"], cmap=IPP_CMAP,
            s=IPP_MARKER_SIZE, alpha=IPP_ALPHA,
            edgecolors="none", transform=proj, zorder=4,
        )
        # Colorbar pequena e DENTRO do mapa (inset) -- só essa escala fica
        # embutida; temperatura e densidade de raios continuam como estavam.
        cax_ipp = ax.inset_axes(IPP_CBAR_RECT, zorder=7)
        cax_ipp.patch.set_facecolor("white")
        cax_ipp.patch.set_alpha(0.75)
        cbar3 = fig.colorbar(sc_ipp, cax=cax_ipp, orientation="horizontal")
        cbar3.set_label("IPP time (UTC, h)", fontsize=12, labelpad=2)
        t_lo, t_hi = sc_ipp.get_clim()
        t_mid = (t_lo + t_hi) / 2.0
        cbar3.set_ticks([t_lo, t_mid, t_hi])
        cbar3.set_ticklabels([_hours_to_hhmm(t_lo), _hours_to_hhmm(t_mid), _hours_to_hhmm(t_hi)])
        cbar3.ax.tick_params(labelsize=12, length=2)
        cbar3.outline.set_linewidth(0.6)
        _label_bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
        cbar3.ax.xaxis.label.set_bbox(_label_bbox)
        for _tl in cbar3.ax.get_xticklabels():
            _tl.set_bbox(_label_bbox)

    if has_epoch:
        ax.scatter(
            ipp_epoch[IPP_LON_COL], ipp_epoch[IPP_LAT_COL],
            marker=IPP_EPOCH_MARKER_STYLE, s=IPP_EPOCH_MARKER_SIZE,
            facecolors=IPP_EPOCH_COLOR, edgecolors=IPP_EPOCH_EDGE_COLOR,
            linewidths=IPP_EPOCH_EDGE_WIDTH, transform=proj, zorder=6,
            label="IPP @ epoch",
        )
        ax.legend(loc="lower left", fontsize=12, framealpha=0.8)

    gl = ax.gridlines(
        draw_labels=True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(5)
    gl.ylocator = mticker.MultipleLocator(5)

    ax.set_title(
        rf"\textbf{{{goes_dt.strftime('%H:%M')} UTC - {goes_dt.strftime('%Y/%m/%d')} GOES Infrared (10.3 $\mu$m)}}"
    )


    # add_provenance_footer(fig, storage_path=str(IMAGE_DIR) + os.sep)

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
def _process_day(pq_path, file_date, ipp_time_range, ipp_pairs, target_time, sess, frame_time_range=None):
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
        return

    # ── filtro de janela de horário (tbeg/tend, usado no modo manual_runs) ──
    if frame_time_range:
        t0, t1 = frame_time_range
        before = len(day_names)
        day_names = [
            n for n in day_names
            if (d := _dt_from_name(n)) is not None
            and t0 <= (d.hour + d.minute / 60.0 + d.second / 3600.0) <= t1
        ]
        print(f"   Filtro de janela ativo: {_hours_to_hhmm(t0)}–{_hours_to_hhmm(t1)} UTC → "
              f"{len(day_names)} de {before} frame(s) mantidos")

        if not day_names:
            print(f"   [WARN] No GOES frames found in {_hours_to_hhmm(t0)}–{_hours_to_hhmm(t1)} for {file_date}")
            return

    # ── filtro de horário: mantém só o frame mais próximo de target_time ──
    if target_time:
        target_hh, target_mm = int(target_time[:2]), int(target_time[2:])
        target_dt = datetime.combine(file_date, datetime.min.time()).replace(
            hour=target_hh, minute=target_mm
        )
        named_dts = [(n, _dt_from_name(n)) for n in day_names]
        named_dts = [(n, d) for n, d in named_dts if d is not None]
        best_name, best_dt = min(
            named_dts, key=lambda nd: abs((nd[1] - target_dt).total_seconds())
        )
        day_names = [best_name]
        print(f"   Filtro de horário ativo: {target_time} → frame mais próximo: "
              f"{best_dt.strftime('%H:%M')} ({best_name})")

    # ── carrega a trajetória de IPPs do dia (uma vez, reaproveitada em todos os frames) ──
    ipp_day = pd.DataFrame()
    if PLOT_IPP_TRAJECTORY:
        doy_of_year = file_date.timetuple().tm_yday
        ipp_day = load_ipp_trajectory(
            year=file_date.year, doy=doy_of_year,
            lat_range=IPP_LAT_RANGE, lon_range=IPP_LON_RANGE,
            time_range=ipp_time_range, stations=IPP_STATIONS, pairs=ipp_pairs,
        )
        print(f"   IPPs (GNSS): {len(ipp_day):,} ponto(s) no ROI/janela definidos")

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

            ipp_epoch = pd.DataFrame()
            if PLOT_IPP_TRAJECTORY and IPP_EPOCH_MARKER:
                ipp_epoch = _nearest_ipp_epoch(
                    ipp_day, goes_dt, max_gap_min=IPP_EPOCH_MAX_GAP_MIN
                )

            plot_goes_lightning(
                goes_dt, temp, lat, lon, nearby, out_path,
                ipp=ipp_day, ipp_epoch=ipp_epoch,
            )

        except Exception as e:
            print(f"   [ERR] Processing {name}: {e}")

        finally:
            try:
                if nc_path.exists():
                    nc_path.unlink(missing_ok=True)
            except Exception:
                pass


sess = _build_session()

if manual_runs:
    print(f"Modo manual_runs ativo: {len(manual_runs)} run(s) configurado(s).")
    for run_cfg in manual_runs:
        year, doy = run_cfg["year"], run_cfg["doy"]
        file_date = datetime(year, 1, 1).date() + timedelta(days=doy - 1)
        date_str  = file_date.strftime("%Y%m%d")

        pq_candidates = sorted(PARQUET_DIR.glob(f"*{date_str}*_{NEW_SAMPLING_RATE}s.parquet"))
        if not pq_candidates:
            print(f"   [WARN] Nenhum parquet de raios encontrado para {date_str} em {PARQUET_DIR}")
            continue

        _process_day(
            pq_candidates[0], file_date,
            ipp_time_range=(run_cfg["tbeg"], run_cfg["tend"]),
            ipp_pairs=run_cfg["pairs"],
            target_time=None,
            sess=sess,
            frame_time_range=(run_cfg["tbeg"], run_cfg["tend"]),
        )
else:
    parquet_files = sorted(PARQUET_DIR.glob(f"*_{NEW_SAMPLING_RATE}s.parquet"))
    parquet_files = _filter_by_date(parquet_files)

    if not parquet_files:
        print(f"No parquet files found in {PARQUET_DIR} (verifique TARGET_DATE / DATE_RANGE)")
    else:
        for pq_path in parquet_files:
            m = re.search(r"(\d{8})", pq_path.name)
            if not m:
                print(f"Skipping {pq_path.name} — cannot parse date")
                continue
            file_date = datetime.strptime(m.group(1), "%Y%m%d").date()
            _process_day(
                pq_path, file_date,
                ipp_time_range=IPP_TIME_RANGE,
                ipp_pairs=IPP_PAIRS,
                target_time=TARGET_TIME,
                sess=sess,
            )

print("\nAll done.")