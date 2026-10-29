"""
plot_goes_keogram_nolightning.py

Rotina ALTERNATIVA ao lightning_goes_trajec_mask_v4.py para os dias SEM dados de raios
(sem .parquet de LFR). Não lê raio nenhum: baixa o GOES (canal 13) da CPTEC, lê as
máscaras de clusters (.npz de detect_clusters_area6) e os keogramas de dTEC já gerados,
e salva UMA figura por frame GOES com DOIS painéis:

    esquerda : mapa GOES IR + isoterma 'lime' da máscara + id de cada cluster
    direita  : keograma(s) de dTEC (lon e/ou lat, empilhados, eixo de tempo compartilhado),
               com a janela do frame (área transparente) e a marcação da TEMPESTADE.

Tracking das tempestades (independe de raios):
    para cada frame, um retângulo por cluster (os que tocam o ROI) com a largura da janela
    do frame e a extensão do cluster em lat/lon (painel de lat/lon). Acumula frame a frame,
    então os retângulos vizinhos se emendam e mostram onde a tempestade esteve ao longo do
    tempo. Basta existir a máscara do frame -- NÃO é exigido nenhum raio dentro do cluster
    (o filtro ONLY_CLUSTERS_WITH_STRIKES da rotina original não existe aqui).

Folder layout:
    data/
        goes_tmp/   ← normalmente VAZIA: o .nc do GOES é baixado para a memória e decodificado com numpy
                      (GOES_IN_MEMORY); só é usada como reserva (arquivo que não é NetCDF-3 e sem leitor
                      de biblioteca em memória) ou se GOES_IN_MEMORY = False
        maps_keo/   ← saída: maps_keo/{YYYYMMDD}/*_MAPKEO.png
    (máscaras: MASK_DIR; keogramas: dtec_keogram_dir(year, doy) de shared.config)

Basta ajustar o bloco CONFIG abaixo e rodar (Spyder ou terminal).
"""

# ── LaTeX / cache FORA do C: ─────────────────────────────────────────────────
# Com USE_LATEX = True o matplotlib chama latex + dvipng e grava .tex/.dvi/.png em
# ~/.matplotlib/tex.cache e em %TEMP% (normalmente no C:); o MiKTeX/TeX Live ainda grava
# logs próprios a cada chamada. CACHE_DIR leva o cache do matplotlib e o TEMP para outra pasta.
#   ATENÇÃO: este bloco precisa rodar ANTES do 1º import do matplotlib no processo. No Spyder,
#   reinicie o kernel (ou rode no terminal) depois de mudar; num console que já importou o
#   matplotlib o redirecionamento não tem efeito. Na 1ª execução o cache de fontes é refeito.
USE_LATEX = False           # False = mathtext do matplotlib: sem latex/dvipng, quase nenhuma escrita em
                            #         disco (a fonte do texto muda um pouco; fórmulas ficam em estilo CM).
                            # True  = LaTeX (acabamento de publicação; lento e grava muito no C:)
CACHE_DIR = r"data\cache"   # relativo à pasta de execução (a mesma do data\goes_tmp); None = não redireciona

import os
import tempfile
from pathlib import Path

if CACHE_DIR:
    _cache_root = Path(CACHE_DIR).resolve()
    (_cache_root / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_cache_root / "mpl")     # cache do matplotlib (tex.cache, fontes)
    os.environ["TEMP"] = os.environ["TMP"] = str(_cache_root / "tmp")   # herdado pelo latex/dvipng
    tempfile.tempdir = str(_cache_root / "tmp")

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
mpl.rcParams["font.size"]       = 16
mpl.rcParams["axes.titlesize"]  = 16
mpl.rcParams["axes.labelsize"]  = 16
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

mpl.rcParams["text.usetex"]  = USE_LATEX
mpl.rcParams["font.family"]  = "serif"
if not USE_LATEX:
    mpl.rcParams["mathtext.fontset"] = "cm"   # $\mu$, $T_b$ etc. em Computer Modern (parecido com o LaTeX)

import io
import re
import time
import inspect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# módulo compartilhado do projeto (mesmo usado nas outras rotinas)
from shared.config import dtec_keogram_dir


# ── CONFIG ────────────────────────────────────────────────────────────────────
GOES_IN_MEMORY    = True      # True = baixa o .nc do GOES para a memória e o decodifica com numpy (se for
                              #        NetCDF-3; ver _read_nc_memory) -- nada em disco, ~1 arquivo .nc de RAM
                              #        por vez. False = grava em GOES_TMP_DIR, lê e apaga.
GOES_TMP_DIR      = r"data\goes_tmp"      # só usada com GOES_IN_MEMORY = False, ou como reserva se o .nc NÃO
                                          # for NetCDF-3 e nenhum leitor de biblioteca em memória funcionar
OUT_DIR           = r"data\maps_keo"      # figuras: OUT_DIR/{YYYYMMDD}/*_MAPKEO.png
SKIP_EXISTING     = True      # True = se o PNG do frame já existe em OUT_DIR, pula (nem baixa o GOES). False = refaz
                              #        e sobrescreve tudo. Para refazer só alguns frames, apague os PNGs deles.
CHANNEL           = 13
GOES_TIMEOUT      = 30.0
VMIN_TEMP         = -80       # °C
VMAX_TEMP         = 40        # °C (a escala realçada vai de -80 a 40 °C)

# Escala realçada do canal 13 (10,35 µm): faixas coloridas de 10 °C entre
# -80 e -30 °C e degradê de cinza claro -> preto de -30 a 40 °C.
_TB_PONTOS = [
    (-80, (224, 131, 224)), (-70, (255, 209, 255)),  # magenta -> rosa claro
    (-70, (0, 0, 255)),     (-60, (82, 69, 255)),    # azul
    (-60, (47, 218, 255)),  (-50, (156, 227, 245)),  # ciano
    (-50, (255, 255, 97)),  (-40, (235, 255, 122)),  # amarelo
    (-40, (255, 101, 1)),   (-30, (255, 158, 70)),   # laranja
    (-30, (176, 176, 176)), (40, (0, 0, 0)),         # cinza claro -> preto
]
CMAP_GOES = mpl.colors.LinearSegmentedColormap.from_list(
    "tb_realcada",
    [((t - VMIN_TEMP) / (VMAX_TEMP - VMIN_TEMP), [v / 255 for v in c])
     for t, c in _TB_PONTOS],
    N=1024,
)

# ── seleção de data (sem parquet, então o dia vem sempre daqui) ──────────────
# TARGET_DATE : um dia específico, 'YYYYMMDD'.
# DATE_RANGE  : intervalo (início, fim) 'YYYYMMDD', ambos incluídos. Ignorado se TARGET_DATE estiver definido.
TARGET_DATE       = "20231213"

#Weak Thunderstorms:
TARGET_DATE       = "20240803"
TARGET_DATE       = "20240805"
TARGET_DATE       = "20240811"
TARGET_DATE       = "20240812"
TARGET_DATE       = "20240813"
TARGET_DATE       = "20240815"
TARGET_DATE       = "20240816"
TARGET_DATE       = "20240818"
TARGET_DATE       = "20240819"
TARGET_DATE       = "20240826"
TARGET_DATE       = "20240828"
TARGET_DATE       = "20240829"
# TARGET_DATE       = "20240902"
# TARGET_DATE       = "20240903"
# TARGET_DATE       = "20240907"
# TARGET_DATE       = "20240916"
# TARGET_DATE       = "20260615"
# TARGET_DATE       = "20260616"
# TARGET_DATE       = "20260621"
# TARGET_DATE       = "20260623"
# TARGET_DATE       = "20260627"
DATE_RANGE        = None                 # ex: ("20250110", "20250120")
# TARGET_TIME : 'HHMM' -> só o frame GOES mais próximo desse horário, em cada dia. None = todos os frames.
TARGET_TIME       = None                 # ex: "1230"
# Janela de horário dos frames (horas decimais UTC). None = o dia inteiro.
FRAME_TIME_RANGE          = None         # ex: (15, 19) = só os frames entre 15:00 e 19:00
FRAME_TIME_RANGE_BY_DATE  = {            # sobrescreve FRAME_TIME_RANGE em dias específicos
    # "20260722": (16, 21),
}

# ── mapa (cartopy) ─────────────────────────────────────────────────────────
SHOW_STATES       = True      # fronteiras estaduais (admin_1)
SHOW_COUNTRIES    = True      # fronteiras de países
SHOW_COASTLINE    = True      # linha de costa
MAP_RESOLUTION    = "50m"     # '110m' (rápido/grosseiro), '50m' (equilíbrio), '10m' (detalhado/lento)
STATE_LINEWIDTH   = 0.6
STATE_COLOR       = "dimgray"
BORDER_LINEWIDTH  = 0.8
BORDER_COLOR      = "black"

# ── recorte espacial (ROI) ────────────────────────────────────────────────
CROP_TO_ROI       = True
ROI_LAT_MIN       = -55
ROI_LAT_MAX       = 10
ROI_LON_MIN       = -85
ROI_LON_MAX       = -30

# ── máscaras de clusters (saída de detect_clusters_area6 / save_labels_npz) ──
# <mask_dir>/<nome_do_csv>_labels.npz (um por timestamp do GOES; só existe se houve >=1 cluster):
#   labels (int16, ny×nx; 0 = fora, k = cluster k), lats/lons (centros das células),
#   time_string, t_threshold, ...
# Aqui servem para (1) a ISOTERMA do cluster no mapa e (2) o tracking nos keogramas.
GRID_MASK_DIR  = r"D:\goes-data-pipeline\grid_masks"
GRID_MASK_NAME = "mask_SouthAmerica.npz"
GRID_LAT_KEYS  = ("lats", "lat", "latitude", "latitudes")
GRID_LON_KEYS  = ("lons", "lon", "longitude", "longitudes")

MASK_DIR = r"D:\goes-data-pipeline\grid_masks"
MASK_FILE_GLOB = "*_labels.npz"   # se nada casar, tenta *.npz
# Como o horário do frame aparece no NOME do .npz (testados em ordem; dígitos apenas).
MASK_NAME_PATTERNS = (
    "{Y}{m}{d}{H}{M}",      # 202110231530
    "{Y}{m}{d}_{H}{M}",     # 20211023_1530
    "{Y}{m}{d}T{H}{M}",     # 20211023T1530
    "{Y}{j}{H}{M}",         # 2021296 1530 (ano + DOY)
)
MASK_LABELS_KEYS = ("labels", "labels_filtered", "mask")

# ── mapa: isoterma + id dos clusters ────────────────────────────────────────
PLOT_ISOTHERM      = True
PLOT_CLUSTER_IDS   = True
ISOTHERM_COLOR     = "lime"
ISOTHERM_LINEWIDTH = 1.4
CLUSTER_ID_LABEL_DLAT = 1.0   # deslocamento (graus de latitude, p/ cima) do número do cluster no mapa
# Todos os clusters da máscara que TOCAM o ROI são plotados/marcados (não há filtro de raios).

# ── janela do frame / eixo de tempo ─────────────────────────────────────────
FRAME_HALF_WINDOW_MIN = 5     # meia-largura da janela de cada frame (min): epoch ± 5.
                              # None = metade do passo entre frames (5 min p/ passo de 10 min).
KEO_FIXED_XAXIS   = True      # True = eixo x fixo (1º frame − janela → último frame + janela);
                              # False = o eixo termina na janela do frame atual (o keograma "aparece" aos poucos)
WINDOW_COLOR      = "red"     # janela do frame atual: ÁREA transparente (sem linha no epoch)
WINDOW_ALPHA      = 0.35

# Escala de brightness temperature: pequena, DENTRO do mapa.
TB_CBAR_CORNER     = "upper right"          # "upper right" ou "upper left"
TB_CBAR_SIZE       = (0.36, 0.025)          # (largura, altura) da barra, em fração do mapa
TB_CBAR_TICKS      = (-80, -60, -40, -20, 0, 20, 40)
TB_CBAR_BG_PAD_PTS = 2                      # folga (pt) da caixa branca ao redor de barra + ticks + rótulo

# ── keograma de dTEC (grids .nc gerados por visualize_keo_datashader.py) ─────
# Esta rotina NÃO gera keograma: só LÊ os grids que main_build() já gravou em
#   <dtec_keogram_dir(year, doy)>/keogram_{lat|lon}_{year}_{doy:03d}_{resolution}.nc
# Se um deles não existir, o painel é omitido e o console mostra o main_build(...) para gerá-lo.
PLOT_KEOGRAM      = True
KEO_AXES          = ("lon", "lat")   # de cima p/ baixo; ("lon",), ("lat",)
KEO_RESOLUTION    = "ultra"          # low | high | veryhigh | ultra (o do main_build)
KEO_DTEC_COL      = "dtec_15min"     # coluna do main_build (só p/ o rótulo e a mensagem de aviso)
KEO_TRH           = 0.15             # escala de cor simétrica ±TRH (TECU)
KEO_CMAP          = "seismic_r"
KEO_SMOOTH_SIZE   = 20               # média móvel vertical em px (ignora NaN/zeros); None = sem suavização
KEO_SHOW_CAPTION  = True             # "dTEC keogram (15-min moving average)" acima do 1º keograma
KEO_CBAR_WIDTH    = 0.12             # largura da escala de dTEC (fração da largura da figura)

# Tracking da TEMPESTADE nos keogramas (ver docstring). Frame atual = cor do cluster (a do mapa) com
# borda mais grossa; frames anteriores = uma cor única e fraca (o id do cluster é atribuído frame a
# frame, então a cor não identifica o mesmo sistema entre frames).
KEO_MARK_STORMS     = True
KEO_STORM_ALPHA     = 0.12           # transparência do preenchimento (a borda é opaca)
KEO_STORM_LINEWIDTH = 1.0            # borda dos frames anteriores (a do frame atual é 1,8x)
KEO_STORM_HISTORY_COLOR      = "#b8a2e6"
KEO_STORM_HISTORY_EDGE_ALPHA = 0.7
# O eixo de lat/lon dos keogramas é SEMPRE o extent do mapa da esquerda (ROI_*, ou a extensão do GOES
# se CROP_TO_ROI = False): se o keograma cobrir só parte dessa faixa, o resto fica em branco.

# ── log no console ──────────────────────────────────────────────────────────
LOG_VERBOSE        = False
# ─────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("OMP_NUM_THREADS", "1")

GOES_TMP_DIR  = Path(GOES_TMP_DIR)
OUT_DIR       = Path(OUT_DIR)
MASK_DIR      = Path(MASK_DIR)
GRID_MASK_DIR = Path(GRID_MASK_DIR)


# ── proveniência automática (script / função / storage) ───────────────────────
def add_provenance_footer(fig, storage_path, x=0.9, y=0.01, **kwargs):
    """Rodapé com Script / Function / Storage, descobertos via inspect (sem ficar dessincronizado)."""
    caller = inspect.stack()[1]
    script_name   = os.path.relpath(caller.filename)
    function_name = caller.function
    storage_text  = Path(storage_path).as_posix()  # '/' em vez de '\' (no LaTeX, '\' inicia um comando)

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
        usetex=False,  # o rodapé tem '_' e caminhos; nunca deve passar pelo LaTeX
    )
    defaults.update(kwargs)
    fig.text(x, y, text, **defaults)


def _hours_to_hhmm(h: float) -> str:
    """Hora decimal (ex.: 14.5) -> 'HH:MM' (ex.: '14:30')."""
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        mm = 0
        hh += 1
    hh %= 24
    return f"{hh:02d}:{mm:02d}"


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
    s.headers.update({"User-Agent": "goes-keogram-plot/1.0"})
    return s


# ── list .nc files on server (com cache por mês: DATE_RANGE não relista a mesma pasta) ──
_NC_LIST_CACHE = {}


def _list_nc_files(sess: requests.Session, url: str) -> list:
    if url in _NC_LIST_CACHE:
        return _NC_LIST_CACHE[url]
    try:
        r = sess.get(url, timeout=GOES_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = [
            a["href"] for a in soup.find_all("a", href=True)
            if a["href"].endswith(".nc") and not a["href"].startswith("?")
        ]
        if out:
            _NC_LIST_CACHE[url] = out
        return out
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


# ── download one .nc EM MEMÓRIA (nada é gravado em disco) ───────────────────
def _download_nc_bytes(sess: requests.Session, url: str, nc_name: str) -> bytes | None:
    """
    Baixa o .nc do GOES direto para a memória. Confere o tamanho recebido contra o Content-Length
    (um download cortado no meio devolveria um arquivo truncado, sem o .part/os.replace de antes).
    Devolve os bytes, ou None em caso de erro.
    """
    try:
        buf = io.BytesIO()
        with sess.get(url, stream=True, timeout=GOES_TIMEOUT) as r:
            r.raise_for_status()
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    buf.write(chunk)
            expected = r.headers.get("Content-Length")
            if (expected and expected.isdigit() and not r.headers.get("Content-Encoding")
                    and buf.tell() != int(expected)):
                raise IOError(f"download incompleto: {buf.tell()} de {expected} bytes")
        return buf.getvalue()
    except Exception as e:
        print(f"   [ERR] Downloading {nc_name}: {e}")
        return None


# ── read .nc → arrays ────────────────────────────────────────────────────────
def _dataset_to_arrays(ds) -> tuple:
    lats = ds["lat"].values.astype("float32")
    lons = ds["lon"].values.astype("float32")
    temp = ds["Band1"].astype("float32").values / 100.0 - np.float32(273.13)
    return temp, lats, lons


def _read_nc(nc_path: Path) -> tuple:
    """Leitura a partir de um ARQUIVO (usada só na reserva em disco)."""
    import xarray as xr
    with xr.open_dataset(nc_path) as ds:
        return _dataset_to_arrays(ds)


def _open_nc_memory(raw: bytes):
    """
    Abre o .nc a partir dos bytes, sem arquivo. Tenta, em ordem: netCDF4 em memória (mesma
    decodificação de abrir por caminho), h5netcdf e scipy. Levanta RuntimeError se nenhum servir.
    """
    import xarray as xr
    errors = []
    try:
        import netCDF4 as nc4
        from xarray.backends import NetCDF4DataStore
        return xr.open_dataset(NetCDF4DataStore(nc4.Dataset("goes_in_memory.nc", mode="r", memory=raw)))
    except Exception as e:
        errors.append(f"netCDF4 ({type(e).__name__}: {str(e)[:50]})")
    for engine in ("h5netcdf", "scipy"):
        try:
            return xr.open_dataset(io.BytesIO(raw), engine=engine)
        except Exception as e:
            errors.append(f"{engine} ({type(e).__name__}: {str(e)[:50]})")
    raise RuntimeError("nenhum leitor em memória funcionou: " + "; ".join(errors))


# ── decodificador NetCDF-3 SÓ com numpy/struct (sem xarray, netCDF4, h5py nem scipy) ─────────────
# Se o .nc começa com b"CDF\x01" (clássico) ou b"CDF\x02" (64-bit offset), os bytes baixados são
# decodificados aqui mesmo, em memória. NetCDF-4/HDF5 (b"\x89HDF") NÃO tem como ser lido só com
# numpy (o HDF5 é um contêiner complexo): nesse caso _read_nc_memory cai nos leitores de biblioteca.
_NC3_DTYPES = {1: ">i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8"}   # nc_type -> dtype big-endian


def _parse_nc3_header(raw: bytes) -> dict:
    """Cabeçalho do NetCDF-3: dimensões, atributos globais e, por variável, dims/atributos/tipo/offset."""
    import struct
    if raw[:3] != b"CDF" or raw[3:4] not in (b"\x01", b"\x02"):
        raise ValueError(f"não é NetCDF-3 clássico (começa com {bytes(raw[:4])!r})")
    off64 = raw[3:4] == b"\x02"
    pos = 4

    def u32():
        nonlocal pos
        v = struct.unpack_from(">I", raw, pos)[0]
        pos += 4
        return v

    def u64():
        nonlocal pos
        v = struct.unpack_from(">Q", raw, pos)[0]
        pos += 8
        return v

    def name():
        nonlocal pos
        n = u32()
        s = bytes(raw[pos:pos + n]).decode("utf-8", "replace")
        pos += (n + 3) & ~3                       # nomes são preenchidos até múltiplo de 4 bytes
        return s

    def attrs():
        nonlocal pos
        tag, n = u32(), u32()
        if tag == 0 and n == 0:                   # lista ausente
            return {}
        if tag != 0x0C:
            raise ValueError("lista de atributos inválida")
        out = {}
        for _ in range(n):
            key, nctype, cnt = name(), u32(), u32()
            dt = np.dtype(_NC3_DTYPES[nctype])
            nbytes = cnt * dt.itemsize
            chunk = bytes(raw[pos:pos + nbytes])
            pos += (nbytes + 3) & ~3
            out[key] = (chunk.decode("utf-8", "replace").rstrip("\x00") if nctype == 2
                        else np.frombuffer(chunk, dtype=dt))
        return out

    u32()                                          # numrecs (só importa p/ variáveis de registro)
    tag, n = u32(), u32()
    dims = []
    if not (tag == 0 and n == 0):
        if tag != 0x0A:
            raise ValueError("lista de dimensões inválida")
        dims = [(name(), u32()) for _ in range(n)]
    gatts = attrs()
    tag, n = u32(), u32()
    variables = {}
    if not (tag == 0 and n == 0):
        if tag != 0x0B:
            raise ValueError("lista de variáveis inválida")
        for _ in range(n):
            vname = name()
            dimids = [u32() for _ in range(u32())]
            vatts = attrs()
            nctype, _vsize = u32(), u32()
            begin = u64() if off64 else u32()
            variables[vname] = dict(dims=[dims[i] for i in dimids], attrs=vatts,
                                    nctype=nctype, begin=begin)
    return dict(dims=dims, gatts=gatts, vars=variables)


def _nc3_read_var(raw: bytes, hdr: dict, vname: str) -> tuple:
    """(array big-endian bruto, atributos) de uma variável NÃO-registro, como uma visão sobre os bytes."""
    if vname not in hdr["vars"]:
        raise KeyError(f"variável '{vname}' não existe no .nc (há: {list(hdr['vars'])})")
    v = hdr["vars"][vname]
    shape = tuple(length for _, length in v["dims"])
    if any(length == 0 for length in shape):
        raise NotImplementedError(f"'{vname}' tem dimensão ilimitada (variável de registro)")
    count = int(np.prod(shape)) if shape else 1
    arr = np.frombuffer(raw, dtype=np.dtype(_NC3_DTYPES[v["nctype"]]), count=count, offset=v["begin"])
    return arr.reshape(shape), v["attrs"]


def _nc3_decode(arr: np.ndarray, attrs: dict) -> np.ndarray:
    """Decodificação CF no estilo do xarray: _Unsigned, _FillValue/missing_value -> NaN, scale_factor/add_offset."""
    a = arr.astype(arr.dtype.newbyteorder("="))    # big-endian -> nativo (cópia independente dos bytes)
    fills = [np.asarray(attrs[k]).ravel() for k in ("_FillValue", "missing_value") if k in attrs]
    if a.dtype.kind == "i" and str(attrs.get("_Unsigned", "")).lower() == "true":
        udt = np.dtype(f"u{a.dtype.itemsize}")
        fills = [f.astype(a.dtype).view(udt) for f in fills]
        a = a.view(udt)
    mask = None
    for f in fills:
        for fv in f:
            m = a == fv
            mask = m if mask is None else (mask | m)
    scale, offset = attrs.get("scale_factor"), attrs.get("add_offset")
    if a.dtype.kind in "iu" and (mask is not None or scale is not None or offset is not None):
        a = a.astype(np.float32 if a.dtype.itemsize <= 2 else np.float64)
    if scale is not None:
        a = a * np.asarray(scale).ravel()[0]
    if offset is not None:
        a = a + np.asarray(offset).ravel()[0]
    if mask is not None:
        a = np.where(mask, np.nan, a)
    return a


def _read_nc3_numpy(raw: bytes) -> tuple:
    """(temp °C, lats, lons) a partir dos bytes de um NetCDF-3 -- mesma conta do _dataset_to_arrays."""
    hdr = _parse_nc3_header(raw)

    def get(vname):
        arr, attrs = _nc3_read_var(raw, hdr, vname)
        return _nc3_decode(arr, attrs)

    lats = get("lat").astype("float32")
    lons = get("lon").astype("float32")
    temp = get("Band1").astype("float32") / 100.0 - np.float32(273.13)
    return temp, lats, lons


_READER_NAME = None      # qual leitor decodificou o último frame (para o log)
_READER_PRINTED = None


def _read_nc_memory(raw: bytes) -> tuple:
    """
    1) NetCDF-3 (CDF1/CDF2): decodificado só com numpy. 2) Qualquer outro caso (NetCDF-4/HDF5, ou
    recurso que o parser mínimo não cobre): leitores de biblioteca, ainda em memória.
    """
    global _READER_NAME
    head = bytes(raw[:4])
    if head[:3] == b"CDF" and head[3:4] in (b"\x01", b"\x02"):
        try:
            out = _read_nc3_numpy(raw)
            _READER_NAME = "numpy (NetCDF-3, sem biblioteca)"
            return out
        except NotImplementedError:
            pass   # ex.: variável de registro -> tenta as bibliotecas
    try:
        with _open_nc_memory(raw) as ds:
            out = _dataset_to_arrays(ds)
    except Exception as e:
        if head == b"\x89HDF":
            raise RuntimeError("o .nc é NetCDF-4/HDF5: não dá para ler só com numpy; precisa de netCDF4 "
                               f"ou h5netcdf+h5py ({type(e).__name__}: {str(e)[:70]})") from e
        raise
    _READER_NAME = "biblioteca (xarray)"
    return out


def _log_reader_once():
    global _READER_PRINTED
    if _READER_NAME != _READER_PRINTED:
        print(f"   leitor do .nc: {_READER_NAME}")
        _READER_PRINTED = _READER_NAME


_MEMORY_READ_OK = None   # None = ainda não testado; True/False depois do 1º frame


def _read_goes(raw: bytes, nc_name: str) -> tuple:
    """
    Lê o frame a partir dos bytes (memória). Se GOES_IN_MEMORY = False, ou se o 1º frame mostrar
    que nenhum leitor em memória funciona neste ambiente (e o disco funciona), cai para um arquivo
    temporário em GOES_TMP_DIR (gravado, lido e apagado) -- o comportamento antigo.
    """
    global _MEMORY_READ_OK
    mem_error = None
    if GOES_IN_MEMORY and _MEMORY_READ_OK is not False:
        try:
            out = _read_nc_memory(raw)
            _MEMORY_READ_OK = True
            _log_reader_once()
            return out
        except Exception as e:
            if _MEMORY_READ_OK:   # já funcionou antes: o problema é ESTE arquivo, não o leitor
                raise
            mem_error = e

    GOES_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_nc = GOES_TMP_DIR / nc_name
    try:
        tmp_nc.write_bytes(raw)
        out = _read_nc(tmp_nc)
    finally:
        try:
            tmp_nc.unlink(missing_ok=True)
        except Exception:
            pass
    if mem_error is not None:   # o disco leu o que a memória não leu -> o problema é o leitor, não o arquivo
        _MEMORY_READ_OK = False
        print(f"   [WARN] leitura em memória indisponível ({str(mem_error)[:140]}); "
              f"seguindo com arquivo temporário em {GOES_TMP_DIR}")
    return out


# ── recorte espacial (ROI) ─────────────────────────────────────────────────
def _crop_to_roi(temp: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple:
    """Recorta temperatura/lat/lon para o ROI (grade regular, 1D, como na pasta 'retangular' da CPTEC)."""
    if not CROP_TO_ROI:
        return temp, lat, lon

    lat_mask = (lat >= ROI_LAT_MIN) & (lat <= ROI_LAT_MAX)
    lon_mask = (lon >= ROI_LON_MIN) & (lon <= ROI_LON_MAX)

    if not lat_mask.any() or not lon_mask.any():
        print("   [WARN] ROI fora da área coberta pelo arquivo GOES — usando extensão completa")
        return temp, lat, lon

    temp_cropped = temp[np.ix_(lat_mask, lon_mask)]
    return temp_cropped, lat[lat_mask], lon[lon_mask]


# ── máscaras de clusters (.npz salvos por detect_clusters_area6) ─────────────
_MASK_INDEX = None       # cache: [(Path, só-dígitos do caminho relativo)] de MASK_DIR
_MASK_TS_INDEX = None    # cache: [(Path, dígitos do time_string, time_string como datetime)] (fallback)
_MASK_MISS_LOGGED = 0


def _digits(s) -> str:
    """Só os dígitos: '2021-10-23 15:30' / '20211023_1530' -> '202110231530'."""
    return re.sub(r"\D", "", str(s))


def _mask_time_strings(goes_dt: datetime) -> list:
    """Chaves só-dígitos do horário do frame (a comparação ignora separadores)."""
    fields = dict(
        Y=f"{goes_dt.year:04d}", m=f"{goes_dt.month:02d}", d=f"{goes_dt.day:02d}",
        H=f"{goes_dt.hour:02d}", M=f"{goes_dt.minute:02d}",
        j=f"{goes_dt.timetuple().tm_yday:03d}",
    )
    keys = []
    for p in MASK_NAME_PATTERNS:
        k = _digits(p.format(**fields))
        if k and k not in keys:
            keys.append(k)
    return keys


def _read_time_string(p: Path) -> str | None:
    try:
        with np.load(p, allow_pickle=True) as d:
            if "time_string" in d.files:
                raw = np.asarray(d["time_string"])
                return str(raw.item() if raw.ndim == 0 else raw.ravel()[0])
    except Exception:
        pass
    return None


def _parse_time_string(ts) -> datetime | None:
    """Interpreta o time_string gravado na máscara como data/hora (minuto)."""
    s = str(ts).replace("UTC", "").strip()
    m = re.match(r"\s*((?:19|20)\d{2})\D?(\d{2})\D?(\d{2})\D?(\d{2})\D?(\d{2})", s)
    if m:
        try:
            return datetime(*(int(g) for g in m.groups()))
        except ValueError:
            pass
    for dayfirst in (True, False):
        try:
            t = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        except Exception:
            t = pd.NaT
        if pd.notna(t):
            if t.tzinfo is not None:
                t = t.tz_convert(None)
            return t.to_pydatetime().replace(second=0, microsecond=0)
    return None


def _ensure_mask_index() -> int:
    """Varre MASK_DIR uma única vez e monta o índice dos arquivos de labels."""
    global _MASK_INDEX
    if _MASK_INDEX is not None:
        return len(_MASK_INDEX)

    if not MASK_DIR.exists():
        print(f"   [WARN] MASK_DIR não existe (ou o disco não está acessível): {MASK_DIR}")
        _MASK_INDEX = []
        return 0

    files = sorted(MASK_DIR.rglob(MASK_FILE_GLOB))
    if not files:   # nome fora do padrão <csv>_labels.npz: aceita qualquer .npz de labels
        files = [p for p in sorted(MASK_DIR.rglob("*.npz")) if p.name != GRID_MASK_NAME]
    _MASK_INDEX = [(p, _digits("".join(p.relative_to(MASK_DIR).with_suffix("").parts)))
                   for p in files]

    if LOG_VERBOSE:
        for p in files[:3]:
            print(f"      ex.: {p.relative_to(MASK_DIR)}")
        if files:
            try:
                with np.load(files[0], allow_pickle=True) as d:
                    print(f"      chaves de {files[0].name}: {d.files}; "
                          f"time_string = {_read_time_string(files[0])!r}")
            except Exception as e:
                print(f"      [WARN] não consegui abrir {files[0].name}: {e}")
    return len(files)


def _find_mask_file(goes_dt: datetime) -> Path | None:
    """
    Acha o .npz de labels do frame comparando só os DÍGITOS do horário:
    1) no caminho relativo do arquivo; 2) se nada casar, no 'time_string' gravado DENTRO dos .npz.
    """
    global _MASK_TS_INDEX, _MASK_MISS_LOGGED
    _ensure_mask_index()

    wanted = _mask_time_strings(goes_dt)
    for key in wanted:
        for p, dig in _MASK_INDEX:
            if key in dig:
                return p

    if _MASK_TS_INDEX is None:
        _MASK_TS_INDEX = []
        for p, _ in _MASK_INDEX:
            ts = _read_time_string(p)
            if ts is not None:
                _MASK_TS_INDEX.append((p, _digits(ts), _parse_time_string(ts)))
        if LOG_VERBOSE:
            print(f"   Máscaras: fallback por time_string indexou {len(_MASK_TS_INDEX)} "
                  f"de {len(_MASK_INDEX)} arquivo(s)")

    goes_min = goes_dt.replace(second=0, microsecond=0)
    for p, dig, parsed in _MASK_TS_INDEX:
        if parsed is not None and parsed == goes_min:
            return p
        if any(key in dig for key in wanted):
            return p

    if _MASK_INDEX and _MASK_MISS_LOGGED < 2 and LOG_VERBOSE:
        _MASK_MISS_LOGGED += 1
        print(f"   [DIAG] {goes_dt:%H:%M}: nenhuma máscara casou com {wanted}; "
              f"1º arquivo normalizado = {_MASK_INDEX[0][1]!r}")
    return None


_GRID_CACHE = None   # (lats, lons, caminho) da grid mask; lido uma única vez


def _grid_vectors(npz) -> tuple:
    """Procura os vetores lat/lon num .npz aberto (aceita apelidos de chave) -> (lats, lons) 1D ou (None, None)."""
    def pick(keys, axis):
        k = next((k for k in keys if k in npz.files), None)
        if k is None:
            return None
        v = np.asarray(npz[k])
        if v.ndim == 2:                      # malha 2D -> vetor 1D
            v = v[:, 0] if axis == "lat" else v[0, :]
        return v.astype(float).ravel() if v.ndim == 1 else None

    lats, lons = pick(GRID_LAT_KEYS, "lat"), pick(GRID_LON_KEYS, "lon")
    if lats is None or lons is None:
        return None, None
    return lats, lons


def _load_grid_mask() -> tuple:
    """Lê (uma vez) os vetores lat/lon da grid mask (reserva, se o arquivo de labels não trouxer lat/lon)."""
    global _GRID_CACHE
    if _GRID_CACHE is not None:
        return _GRID_CACHE

    cands = [GRID_MASK_DIR / GRID_MASK_NAME]
    if GRID_MASK_DIR.exists():
        cands += sorted(GRID_MASK_DIR.rglob("*SouthAmerica*.npz"))
    cands.append(Path(GRID_MASK_NAME))

    for p in cands:
        if not p.exists():
            continue
        try:
            with np.load(p, allow_pickle=True) as g:
                lats, lons = _grid_vectors(g)
                keys = list(g.files)
        except Exception as e:
            print(f"   [WARN] não consegui ler a grid mask {p}: {e}")
            continue
        if lats is not None:
            _GRID_CACHE = (lats, lons, p)
            print(f"   Grid mask: {p.name} ({len(lats)}×{len(lons)})")
            return _GRID_CACHE
        print(f"   [WARN] {p} não tem vetores lat/lon (chaves: {keys})")

    print("   [WARN] grid mask não encontrada; usando lat/lon dos labels")
    _GRID_CACHE = (None, None, None)
    return _GRID_CACHE


def _load_cluster_mask(goes_dt: datetime) -> dict | None:
    """
    Carrega as labels do frame. Devolve dict(labels, lats, lons, t_threshold, path) ou None se o
    frame não tem máscara (nenhum cluster aprovado nesse horário).
    """
    path = _find_mask_file(goes_dt)
    if path is None:
        return None

    with np.load(path, allow_pickle=True) as d:
        key = next((k for k in MASK_LABELS_KEYS if k in d.files), None)
        if key is None:
            key = next((k for k in d.files
                        if d[k].ndim == 2 and (np.issubdtype(d[k].dtype, np.integer)
                                               or d[k].dtype == bool)), None)
            if key is None:
                raise KeyError(f"{path.name}: sem matriz de labels; chaves = {d.files}")
            print(f"   [WARN] chave de labels não reconhecida em {path.name}; usando '{key}' "
                  f"(chaves: {d.files}). Ajuste MASK_LABELS_KEYS se estiver errado.")
        labels = np.asarray(d[key]).astype(np.int32)

        own_lats, own_lons = _grid_vectors(d)
        t_thr = float(d["t_threshold"]) if "t_threshold" in d.files else None

    def _fits(la, lo):
        return la is not None and lo is not None and labels.shape == (len(la), len(lo))

    lats = lons = None
    g_lats = g_lons = g_path = None
    if _fits(own_lats, own_lons):
        lats, lons = own_lats, own_lons
    else:
        g_lats, g_lons, g_path = _load_grid_mask()
        if _fits(g_lats, g_lons):
            lats, lons = g_lats, g_lons

    if lats is None:
        own_shape = None if own_lats is None else (len(own_lats), len(own_lons))
        grid_shape = None if g_lats is None else (len(g_lats), len(g_lons))
        raise ValueError(
            f"{path.name}: labels {labels.shape} não casam com os lat/lon do próprio arquivo "
            f"({own_shape}) nem com a grid mask ({g_path}; shape {grid_shape})")

    return dict(labels=labels, lats=lats, lons=lons, t_threshold=t_thr, path=path)


def _crop_mask_to_roi(mask: dict) -> dict:
    """Recorta as labels para o ROI (mesmo critério de _crop_to_roi): só sobram os clusters que TOCAM o ROI."""
    if not CROP_TO_ROI:
        return mask
    lat_m = (mask["lats"] >= ROI_LAT_MIN) & (mask["lats"] <= ROI_LAT_MAX)
    lon_m = (mask["lons"] >= ROI_LON_MIN) & (mask["lons"] <= ROI_LON_MAX)
    if not lat_m.any() or not lon_m.any():
        return mask
    return dict(
        labels=mask["labels"][np.ix_(lat_m, lon_m)],
        lats=mask["lats"][lat_m], lons=mask["lons"][lon_m],
        t_threshold=mask["t_threshold"], path=mask.get("path"),
    )


# ── feições geográficas + grade ─────────────────────────────────────────────
def _add_map_features(ax):
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


def _add_gridlines(ax):
    gl = ax.gridlines(
        draw_labels=True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(5)
    gl.ylocator = mticker.MultipleLocator(5)


def _bold(s) -> str:
    """Negrito em LaTeX só quando text.usetex está ligado (senão o \\textbf apareceria literal)."""
    return rf"\textbf{{{s}}}" if mpl.rcParams["text.usetex"] else str(s)


def _draw_isotherm(ax, mask: dict, proj, label: bool = True) -> bool:
    """Contorno 'lime' da máscara de clusters (isoterma de t_threshold). Devolve True se desenhou algo."""
    binary = (mask["labels"] > 0).astype(np.float32)
    if not binary.any():
        return False

    ax.contour(
        mask["lons"], mask["lats"], binary, levels=[0.5],
        colors=ISOTHERM_COLOR, linewidths=ISOTHERM_LINEWIDTH,
        transform=proj, zorder=5,
    )
    if label:
        thr = mask.get("t_threshold")
        txt = (rf"Cluster mask ($T_b < {thr:g}\,^{{\circ}}$C)" if thr is not None
               else "Cluster mask")
        ax.plot([], [], color=ISOTHERM_COLOR, linewidth=ISOTHERM_LINEWIDTH,
                label=txt, transform=proj)
    return True


# ── keogramas de dTEC (grids .nc de visualize_keo_datashader.main_build) ─────
_KEO_SPEC = {   # eixo -> (coordenada y dentro do .nc, rótulo do eixo y, faixa do ROI p/ recorte)
    "lat": ("lts", "Latitude (deg)",  (ROI_LAT_MIN, ROI_LAT_MAX)),
    "lon": ("lns", "Longitude (deg)", (ROI_LON_MIN, ROI_LON_MAX)),
}


def _smooth_vertical_nan(arr, size):
    """Média móvel vertical (eixo 0 = graus) ignorando NaN e zeros (bins vazios)."""
    if size is None or size <= 1:
        return arr
    from scipy.ndimage import uniform_filter1d
    arr = arr.astype(float)
    valid = np.isfinite(arr) & (arr != 0.0)
    num = uniform_filter1d(np.where(valid, arr, 0.0), size=size, axis=0, mode="nearest")
    den = uniform_filter1d(valid.astype(float), size=size, axis=0, mode="nearest")
    return np.divide(num, den, out=np.full_like(arr, np.nan), where=den > 0)


def _warn_keograms_missing(year, doy, folder: Path, missing: list):
    """Avisa quais keogramas faltam e como gerá-los (a rotina não gera, só lê)."""
    print("   ⚠ Keograma ainda NÃO gerado -- a figura sai sem este(s) painel(is):")
    for p in missing:
        print(f"      não existe: {p}")
    others = (sorted(p.name for p in folder.glob(f"keogram_*_{year}_{doy:03d}_*.nc"))
              if folder.exists() else [])
    if others:
        print(f"      já existem nesta pasta: {', '.join(others)}  (se for o caso, ajuste KEO_RESOLUTION)")
    print("      → gere com visualize_keo_datashader.py e rode esta rotina de novo:")
    print(f"        main_build(year={year}, doy={doy}, resolution='{KEO_RESOLUTION}', "
          f"dtec_col='{KEO_DTEC_COL}')")


def _load_keogram_panels(file_date, x_range) -> list:
    """
    Lê (uma vez por dia) os keogramas JÁ GERADOS e devolve um painel por eixo de KEO_AXES
    (de cima p/ baixo): dict(axis, ylabel, data (ny, nx), extent [x0, x1, y0, y1] com x em datas do
    matplotlib e y = faixa de lat/lon coberta pelo dado). Recorta ao intervalo de tempo da figura
    (x_range) e ao ROI ANTES de guardar. Arquivo inexistente -> painel omitido + aviso.
    """
    if not (PLOT_KEOGRAM and KEO_AXES and x_range is not None):
        return []
    import xarray as xr

    year, doy = file_date.year, file_date.timetuple().tm_yday
    folder = Path(dtec_keogram_dir(year, doy))
    base_dt = datetime.combine(file_date, datetime.min.time())
    base = pd.Timestamp(base_dt)
    t_lo = (pd.Timestamp(x_range[0]) - base).total_seconds()   # t_seconds = s desde 00:00 UTC do dia
    t_hi = (pd.Timestamp(x_range[1]) - base).total_seconds()

    panels, missing = [], []
    for axis in KEO_AXES:
        ycoord, ylabel, (roi_lo, roi_hi) = _KEO_SPEC[axis]
        path = folder / f"keogram_{axis}_{year}_{doy:03d}_{KEO_RESOLUTION}.nc"
        if not path.exists():
            missing.append(path)
            continue
        try:
            with xr.open_dataarray(path) as da:
                t = np.asarray(da["t_seconds"].values, dtype=float)
                it = np.nonzero((t >= t_lo) & (t <= t_hi))[0]
                if it.size == 0:
                    print(f"   [WARN] {path.name}: sem colunas entre "
                          f"{pd.Timestamp(x_range[0]):%H:%M} e {pd.Timestamp(x_range[1]):%H:%M}")
                    continue
                sub = da.transpose(ycoord, "t_seconds").isel(
                    t_seconds=slice(int(it[0]), int(it[-1]) + 1))
                data = np.asarray(sub.values, dtype=float)            # (ny, nx), só o trecho de tempo
                y = np.asarray(sub[ycoord].values, dtype=float)
                t_sel = t[int(it[0]): int(it[-1]) + 1]
        except Exception as e:   # keograma ilegível não pode derrubar a figura
            print(f"   [WARN] não consegui ler {path.name}: {type(e).__name__}: {str(e)[:60]}")
            continue

        data = _smooth_vertical_nan(data, KEO_SMOOTH_SIZE)           # antes do recorte: sem efeito de borda
        if y[0] > y[-1]:                                             # imshow(origin='lower') quer y crescente
            data, y = data[::-1], y[::-1]
        if CROP_TO_ROI:   # só economiza memória: o eixo y exibido é o extent do mapa
            sel = (y >= roi_lo) & (y <= roi_hi)
            if sel.any():
                data, y = data[sel], y[sel]
        ylim = (float(y.min()), float(y.max()))
        x0 = base_dt + timedelta(seconds=float(t_sel[0]))
        x1 = base_dt + timedelta(seconds=float(t_sel[-1]))
        panels.append(dict(
            axis=axis, ylabel=ylabel, data=data,
            extent=[mdates.date2num(x0), mdates.date2num(x1), ylim[0], ylim[1]],
        ))

    if missing:
        _warn_keograms_missing(year, doy, folder, missing)
    if panels:
        print(f"  keograma: {'+'.join(p['axis'] for p in panels)} ({KEO_RESOLUTION}) · "
              f"{pd.Timestamp(x_range[0]):%H:%M}–{pd.Timestamp(x_range[1]):%H:%M}")
    return panels


def _cluster_extents(mask: dict, only_ids=None) -> dict:
    """
    {id: {"lat": (lo, hi), "lon": (lo, hi)}}: extensão lat/lon de cada cluster da máscara COMPLETA
    (até a borda das células, não só os centros), p/ marcar a tempestade nos keogramas.
    only_ids restringe aos clusters dados (os que tocam o ROI, os mesmos do mapa).
    """
    labels = mask["labels"]
    rows, cols = np.nonzero(labels)
    if rows.size == 0:
        return {}
    ids = labels[rows, cols]
    if only_ids is not None:
        keep = np.isin(ids, np.asarray(only_ids))
        rows, cols, ids = rows[keep], cols[keep], ids[keep]
        if ids.size == 0:
            return {}
    lats, lons = np.asarray(mask["lats"], dtype=float), np.asarray(mask["lons"], dtype=float)
    hl = float(np.median(np.abs(np.diff(lats)))) / 2.0 if len(lats) > 1 else 0.0   # meia célula
    ho = float(np.median(np.abs(np.diff(lons)))) / 2.0 if len(lons) > 1 else 0.0
    df = pd.DataFrame({"id": ids, "lat": lats[rows], "lon": lons[cols]})
    g = df.groupby("id").agg(lat_lo=("lat", "min"), lat_hi=("lat", "max"),
                             lon_lo=("lon", "min"), lon_hi=("lon", "max"))
    return {int(i): dict(lat=(float(r.lat_lo) - hl, float(r.lat_hi) + hl),
                         lon=(float(r.lon_lo) - ho, float(r.lon_hi) + ho))
            for i, r in g.iterrows()}


def _right_column_layout(n_keo: int) -> dict:
    """
    Retângulos [x, y, largura, altura] (fração da figura) da coluna da direita: n_keo keogramas
    empilhados ocupando TODA a altura do mapa. Na faixa acima do 1º keograma ficam o caption (à
    esquerda) e a escala de dTEC (barra horizontal pequena alinhada à direita, KEO_CBAR_WIDTH).
    Com n_keo = 0 devolve 1 retângulo (usado pelo aviso "keograma indisponível").
    """
    y_bot, y_top, gap, gap_top = 0.12, 0.92, 0.03, 0.065
    x_r, w_r = 0.55, 0.43
    n = max(n_keo, 1)
    kh = ((y_top - y_bot) - gap_top - (n - 1) * gap) / n
    rects, top = [], y_top - gap_top          # o 1º keograma começa logo abaixo da faixa do caption
    for _ in range(n):
        rects.append([x_r, top - kh, w_r, kh])
        top -= kh + gap
    return dict(keo=rects,
                cax=[x_r + w_r - KEO_CBAR_WIDTH, y_top - gap_top + 0.005, KEO_CBAR_WIDTH, 0.0125])


def _draw_storm_marks(axk, axis, storm_history, cur_storm, color_of):
    """
    Tracking da tempestade num painel de keograma (axis = 'lat' ou 'lon'): um retângulo por cluster e
    por frame, com a largura da janela do frame (t0..t1) e a extensão do cluster nesse eixo. Desenha
    todos os frames já processados (cronológico): o frame atual (cur_storm) vai por cima, com as cores
    dos clusters (as do mapa) e borda mais grossa; os anteriores em KEO_STORM_HISTORY_COLOR (fraca).
    """
    from matplotlib.patches import Rectangle
    for fr in storm_history:
        is_cur = fr is cur_storm
        xa = mdates.date2num(pd.Timestamp(fr["t0"]).to_pydatetime())
        xb = mdates.date2num(pd.Timestamp(fr["t1"]).to_pydatetime())
        for cid, ext in fr["clusters"].items():
            y0, y1 = ext[axis]
            col = color_of(cid) if is_cur else KEO_STORM_HISTORY_COLOR
            axk.add_patch(Rectangle(
                (xa, y0), xb - xa, y1 - y0,
                facecolor=mpl.colors.to_rgba(col, KEO_STORM_ALPHA),
                edgecolor=mpl.colors.to_rgba(col, 1.0 if is_cur else KEO_STORM_HISTORY_EDGE_ALPHA),
                linewidth=KEO_STORM_LINEWIDTH * (1.8 if is_cur else 1.0), zorder=4,
            ))


def _draw_keogram_panels(fig, axes_keo, cax_keo, panels, frame_window, x_range, map_extent,
                         storm_history=None, cur_storm=None, color_of=None):
    """
    Desenha os painéis de keograma (tempo × lat/lon), empilhados e com o eixo x compartilhado.
    Área transparente (WINDOW_COLOR/WINDOW_ALPHA) = janela do frame atual. Uma única escala de dTEC
    (±KEO_TRH) para todos. Com KEO_MARK_STORMS e storm_history, marca também a tempestade.
    """
    f0, f1 = (pd.Timestamp(t).to_pydatetime() for t in frame_window)
    lon_lo, lon_hi, lat_lo, lat_hi = map_extent
    im = None
    for i, (axk, p) in enumerate(zip(axes_keo, panels)):
        im = axk.imshow(
            p["data"], extent=p["extent"], origin="lower", aspect="auto",
            cmap=KEO_CMAP, vmin=-KEO_TRH, vmax=KEO_TRH, interpolation="nearest",
        )
        axk.xaxis_date()
        # eixo y = MESMO extent do mapa da esquerda (lat no painel de latitude, lon no de longitude)
        axk.set_ylim((lat_lo, lat_hi) if p["axis"] == "lat" else (lon_lo, lon_hi))
        axk.set_ylabel(p["ylabel"])
        axk.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        axk.grid(True, linestyle=":", alpha=0.4)
        # janela do frame: só uma área transparente (sem linhas de borda nem de epoch)
        axk.axvspan(f0, f1, facecolor=WINDOW_COLOR, edgecolor="none", linewidth=0,
                    alpha=WINDOW_ALPHA, zorder=3)
        if KEO_MARK_STORMS and storm_history and color_of is not None:
            _draw_storm_marks(axk, p["axis"], storm_history, cur_storm, color_of)
        if i < len(panels) - 1:
            axk.tick_params(labelbottom=False)   # rótulo de tempo só no painel de baixo

    # eixo de tempo (compartilhado entre os painéis): o imshow mexeria no xlim, então fixa só no fim
    if KEO_FIXED_XAXIS and x_range is not None:
        x0, x1 = (pd.Timestamp(t).to_pydatetime() for t in x_range)
    else:   # termina na janela do frame atual
        x0 = pd.Timestamp(x_range[0]).to_pydatetime() if x_range is not None else f0
        x1 = f1
    axes_keo[0].set_xlim(mdates.date2num(x0), mdates.date2num(x1))
    axes_keo[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes_keo[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    axes_keo[-1].set_xlabel("Time (UTC)")

    m = re.findall(r"\d+", KEO_DTEC_COL)
    mov = f" ({m[0]}-min moving average)" if m else ""
    if KEO_SHOW_CAPTION:   # acima do 1º keograma (fora da imagem: não cobre o dado)
        axes_keo[0].set_title(f"dTEC keogram{mov}", fontsize=11, loc="left", pad=3)
    # escala de dTEC: barra horizontal pequena acima do 1º keograma (ticks em cima, rótulo à esquerda)
    cb = fig.colorbar(im, cax=cax_keo, orientation="horizontal", ticklocation="top")
    cb.set_ticks([-KEO_TRH, 0, KEO_TRH])
    cb.ax.tick_params(labelsize=10, length=2, pad=1)
    cb.outline.set_linewidth(0.6)
    cb.ax.text(-0.04, 0.5, "dTEC (TECU)", transform=cb.ax.transAxes, ha="right", va="center", fontsize=10)


def _add_tb_colorbar(fig, ax, im):
    """
    Escala de brightness temperature PEQUENA dentro do mapa (canto conforme TB_CBAR_CORNER): barra
    horizontal com poucos ticks e o rótulo embaixo, sobre um fundo branco semitransparente justo ao conteúdo.
    """
    from matplotlib.patches import Rectangle
    w, h = TB_CBAR_SIZE
    x0 = 0.96 - w if TB_CBAR_CORNER == "upper right" else 0.04   # folga p/ os rótulos das pontas (-80 e 40)
    y0 = 0.975 - h
    cax = ax.inset_axes([x0, y0, w, h], zorder=7)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks(list(TB_CBAR_TICKS))
    cb.ax.tick_params(labelsize=10, length=2, pad=1)
    cb.outline.set_linewidth(0.6)
    cb.set_label("Brightness Temperature (°C)", fontsize=10, labelpad=1)

    class _FitBackdrop(Rectangle):
        """Fundo que, na hora de desenhar, se ajusta ao tamanho REAL de barra + ticks + rótulo
        (mais a folga TB_CBAR_BG_PAD_PTS). Se o ajuste falhar, vale o retângulo provisório (aproximado)."""
        def draw(self, renderer):
            try:
                bb = cax.get_tightbbox(renderer)
                bb = bb.padded(renderer.points_to_pixels(TB_CBAR_BG_PAD_PTS)).transformed(ax.transAxes.inverted())
                self.set_bounds(bb.x0, bb.y0, bb.width, bb.height)
            except Exception:
                pass
            super().draw(renderer)

    ax.add_patch(_FitBackdrop(
        (x0 - 0.02, y0 - 0.06), w + 0.04, h + 0.06 + 0.008,
        transform=ax.transAxes, facecolor="white", edgecolor="none", alpha=0.75, zorder=6,
    ))
    return cb


# ── figura: mapa (esquerda) + keograma(s) de dTEC (direita) ─────────────────
def plot_map_keogram(goes_dt, temp, lat, lon, mask_roi, frame_window, x_range, out_path,
                     keo_panels=None, storm_history=None, cur_storm=None):
    """
    Esquerda: mapa GOES + isoterma lime + id de cada cluster (só os que tocam o ROI; sem filtro de raios).
    Direita: keograma(s) de dTEC com a janela do frame e o tracking da tempestade (ver _draw_storm_marks).
    mask_roi = None (sem máscara no frame) -> sai a figura sem clusters, para manter a sequência de frames
    completa em animações. keo_panels vazio -> a coluna da direita mostra um aviso no lugar do keograma.
    storm_history = frames já processados (cronológico); cur_storm = entrada do frame atual (ou None).
    """
    if CROP_TO_ROI:
        lon_min, lon_max = ROI_LON_MIN, ROI_LON_MAX
        lat_min, lat_max = ROI_LAT_MIN, ROI_LAT_MAX
    else:
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 8))

    # mapa quadrado (6,4 x 6,4 pol)
    ax = fig.add_axes([0.06, 0.12, 0.40, 0.80], projection=proj)

    n_keo = len(keo_panels) if keo_panels else 0
    lay = _right_column_layout(n_keo)
    axes_keo, cax_keo = [], None
    if n_keo:
        for r in lay["keo"]:
            axes_keo.append(fig.add_axes(r, sharex=axes_keo[0] if axes_keo else None))
        cax_keo = fig.add_axes(lay["cax"])
    else:
        axp = fig.add_axes(lay["keo"][0])
        axp.set_xticks([])
        axp.set_yticks([])
        axp.text(0.5, 0.5, "dTEC keogram not available", ha="center", va="center",
                 transform=axp.transAxes, fontsize=14)

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    im = ax.imshow(
        temp, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max],
        transform=proj, cmap=CMAP_GOES, vmin=VMIN_TEMP, vmax=VMAX_TEMP,
        interpolation="nearest", zorder=1,
    )
    _add_tb_colorbar(fig, ax, im)
    _add_map_features(ax)

    cmap_id = plt.get_cmap("tab10")
    color_of = lambda cid: cmap_id((int(cid) - 1) % 10)

    has_mask = mask_roi is not None and (mask_roi["labels"] > 0).any()
    labels = mask_roi["labels"] if has_mask else None

    if has_mask:
        if PLOT_ISOTHERM:
            _draw_isotherm(ax, mask_roi, proj)

        if PLOT_CLUSTER_IDS:
            # id do cluster no centróide (da parte do cluster dentro do ROI), deslocado
            # CLUSTER_ID_LABEL_DLAT graus p/ cima; limitado a 0,5° abaixo da borda superior do mapa
            for cid in np.unique(labels[labels > 0]):
                rows, cols = np.nonzero(labels == cid)
                ax.text(
                    float(mask_roi["lons"][cols].mean()),
                    min(float(mask_roi["lats"][rows].mean()) + CLUSTER_ID_LABEL_DLAT, lat_max - 0.5),
                    _bold(int(cid)), color="white", fontsize=12,
                    ha="center", va="center", transform=proj, zorder=8,
                    bbox=dict(boxstyle="circle,pad=0.12", fc=color_of(cid), ec="black", lw=0.8),
                )
        if PLOT_ISOTHERM:
            ax.legend(loc="lower left", fontsize=12, framealpha=0.8)
    else:
        note = ("No cluster mask for this frame" if mask_roi is None
                else "No cluster in the ROI for this frame")
        ax.text(0.5, 0.03, note, ha="center", va="bottom", transform=ax.transAxes,
                fontsize=12, zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9))

    _add_gridlines(ax)
    title = (f"{goes_dt.strftime('%H:%M')} UTC - {goes_dt.strftime('%Y/%m/%d')} "
             f"GOES Infrared (10.3 $\\mu$m)")
    if mpl.rcParams["text.usetex"]:
        ax.set_title(rf"\textbf{{{title}}}")
    else:   # mathtext não entende \textbf: negrito pelo peso da fonte
        ax.set_title(title, fontweight="bold")

    if n_keo:
        _draw_keogram_panels(fig, axes_keo, cax_keo, keo_panels, frame_window, x_range,
                             map_extent=(lon_min, lon_max, lat_min, lat_max),
                             storm_history=storm_history, cur_storm=cur_storm, color_of=color_of)

    # x=0.46 = borda direita do mapa: o rodapé fica sob o mapa e não colide com o "Time (UTC)" do keograma
    add_provenance_footer(fig, storage_path=out_path.parent, x=0.46)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # grava num nome temporário e só então renomeia: uma execução interrompida no meio do savefig não
    # deixa um PNG pela metade que o SKIP_EXISTING trataria como figura pronta
    tmp_path = out_path.with_name(out_path.stem + ".part.png")
    try:
        fig.savefig(tmp_path, dpi=130, bbox_inches="tight")
    finally:
        plt.close(fig)
    os.replace(tmp_path, out_path)
    if LOG_VERBOSE:
        print(f"   Saved → {out_path.parent.name}/{out_path.name}")
    return out_path


# ── seleção de datas ─────────────────────────────────────────────────────────
def _dates_to_process() -> list:
    """Sem parquet para varrer, o(s) dia(s) vem(vêm) de TARGET_DATE ou DATE_RANGE."""
    if TARGET_DATE:
        return [datetime.strptime(TARGET_DATE, "%Y%m%d").date()]
    if DATE_RANGE:
        start = datetime.strptime(DATE_RANGE[0], "%Y%m%d").date()
        end   = datetime.strptime(DATE_RANGE[1], "%Y%m%d").date()
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return []


def _fmt_dur(seconds: float) -> str:
    """45.2 -> '45 s'; 133 -> '2 min 13 s'; 3725 -> '1 h 02 min'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


# ── helpers do laço de frames ────────────────────────────────────────────────
def _out_path_for(file_date, goes_dt) -> Path:
    """Caminho do PNG de um frame (o nome só depende do dia e do horário do frame)."""
    doy = goes_dt.timetuple().tm_yday
    out_name = (
        f"{file_date.strftime('%Y%m%d')}_DOY{doy:03d}"
        f"_{goes_dt.strftime('%H%M')}_CH{CHANNEL}_MAPKEO.png"
    )
    return OUT_DIR / file_date.strftime('%Y%m%d') / out_name


def _figure_exists(out_path: Path) -> bool:
    try:
        return out_path.is_file() and out_path.stat().st_size > 0
    except OSError:
        return False


def _frame_mask(goes_dt) -> tuple:
    """
    Máscara de clusters do frame (None = nenhum cluster aprovado nesse horário). Devolve
    (mask_full, mask_roi, ids_roi, n_all, n_act, mask_note): clusters na máscara / clusters que tocam o ROI.
    """
    mask_full, mask_roi = None, None
    ids_roi = np.zeros(0, dtype=np.int32)
    n_all = n_act = 0
    mask_note = "sem máscara"
    try:
        mask_full = _load_cluster_mask(goes_dt)
    except Exception as e:   # máscara ilegível não pode derrubar o mapa
        mask_note = f"⚠ máscara: {type(e).__name__}: {str(e)[:40]}"
        mask_full = None
    if mask_full is not None:
        mask_roi = _crop_mask_to_roi(mask_full)   # só sobram os clusters que tocam o ROI
        ids_roi = np.unique(mask_roi["labels"][mask_roi["labels"] > 0])
        n_all = len(np.unique(mask_full["labels"][mask_full["labels"] > 0]))
        n_act = len(ids_roi)
    return mask_full, mask_roi, ids_roi, n_all, n_act, mask_note


# ── processamento de um dia ──────────────────────────────────────────────────
def _process_day(file_date, frame_time_range, target_time, sess):
    t_day0 = time.perf_counter()
    doy_day = file_date.timetuple().tm_yday
    print(f"\n{'═'*52}")
    print(f"{file_date} (DOY {doy_day:03d}) · sem dados de raios")
    win_txt = tt_txt = ""

    url   = _month_url(file_date.year, file_date.month, CHANNEL)
    names = _list_nc_files(sess, url)
    day_str   = f"_{file_date.year}{file_date.month:02d}{file_date.day:02d}"
    day_names = [n for n in names if day_str in n]

    if not day_names:
        print(f"   [WARN] No GOES files found for {file_date}")
        return

    # ── filtro de janela de horário ──
    if frame_time_range:
        t0, t1 = frame_time_range
        day_names = [
            n for n in day_names
            if (d := _dt_from_name(n)) is not None
            and t0 <= (d.hour + d.minute / 60.0 + d.second / 3600.0) <= t1
        ]
        win_txt = f"{_hours_to_hhmm(t0)}–{_hours_to_hhmm(t1)}"

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
        tt_txt = f"alvo {target_time} → {best_dt:%H:%M}"

    # passo de amostragem da CPTEC a partir dos intervalos entre frames consecutivos
    frame_dts = sorted([_dt_from_name(n) for n in day_names if _dt_from_name(n)])
    if len(frame_dts) >= 2:
        gaps = [(frame_dts[i+1] - frame_dts[i]).seconds // 60
                for i in range(len(frame_dts) - 1)]
        cptec_sampling_min = min(gaps)
    else:
        cptec_sampling_min = 10   # fallback se só houver um frame
    n_frames = len(day_names)
    n_mask_files = _ensure_mask_index()
    partes = [f"{n_frames} frames" + (f" {win_txt}" if win_txt else ""),
              f"passo {cptec_sampling_min} min"]
    if tt_txt:
        partes.append(tt_txt)
    partes.append(f"{n_mask_files} máscaras")
    print("  " + " · ".join(partes))
    print(f"{'─'*52}")
    w_cl = 9

    n_png = n_nomask = n_err = n_skip = 0

    # janela de cada frame: epoch ± frame_half (meia-aberta [t0, t1), para frames vizinhos se emendarem)
    frame_half = timedelta(minutes=(FRAME_HALF_WINDOW_MIN if FRAME_HALF_WINDOW_MIN is not None
                                    else cptec_sampling_min / 2))
    storm_history = []   # extensão lat/lon dos clusters por frame (tracking nos keogramas); zera a cada dia
    x_range = None
    if frame_dts:
        # eixo x: do começo da janela do 1º frame ao fim da janela do último
        x_range = (pd.Timestamp(frame_dts[0] - frame_half), pd.Timestamp(frame_dts[-1] + frame_half))

    # todas as figuras do dia já existem: não há o que fazer (nem carrega o keograma, que é a parte cara)
    if SKIP_EXISTING and frame_dts and all(
            _figure_exists(_out_path_for(file_date, d)) for d in frame_dts):
        print(f"  ✓ as {len(frame_dts)} figuras já existem em {OUT_DIR / file_date.strftime('%Y%m%d')} -> nada a fazer")
        return

    # keogramas de dTEC do dia: lidos UMA vez (só se os .nc já existirem; senão avisa como gerar)
    keo_panels = _load_keogram_panels(file_date, x_range)

    # ordem cronológica (o tracking acumula frame a frame, então a ordem importa)
    for i_frame, name in enumerate(
            sorted(day_names, key=lambda n: _dt_from_name(n) or datetime.min), 1):
        goes_dt = _dt_from_name(name)
        if not goes_dt:
            continue

        t_frame0 = time.perf_counter()
        tag = f"[{i_frame:>{len(str(n_frames))}}/{n_frames}] {goes_dt:%H:%M}"

        out_path = _out_path_for(file_date, goes_dt)
        seg_t0 = pd.Timestamp(goes_dt - frame_half)
        seg_t1 = pd.Timestamp(goes_dt + frame_half)

        if SKIP_EXISTING and _figure_exists(out_path):
            # figura pronta: não baixa o GOES nem plota. Mas o tracking é ACUMULATIVO, então os clusters
            # deste frame entram no histórico (só a máscara, que é leve) -- as figuras seguintes saem
            # iguais às de uma execução completa.
            n_skip += 1
            if keo_panels and KEO_MARK_STORMS:
                try:
                    m_full, _m_roi, m_ids = _frame_mask(goes_dt)[:3]
                    if m_full is not None:
                        storm_history.append(dict(t0=seg_t0, t1=seg_t1,
                                                  clusters=_cluster_extents(m_full, only_ids=m_ids)))
                except Exception:
                    pass
            print(f"  {tag} │ {'já existe':<{w_cl}} │ pulado")
            continue

        raw = _download_nc_bytes(sess, urljoin(url, name), name)
        if raw is None:
            n_err += 1
            print(f"  {tag} │ ✗ download")
            continue

        try:
            temp, lat, lon = _read_goes(raw, name)
            raw = None   # temp/lat/lon são cópias: libera o .nc da memória já
            temp, lat, lon = _crop_to_roi(temp, lat, lon)

            # ── máscara de clusters do frame (None = nenhum cluster aprovado nesse horário) ──
            mask_full, mask_roi, ids_roi, n_all, n_act, mask_note = _frame_mask(goes_dt)
            if mask_full is None:
                n_nomask += 1
            elif LOG_VERBOSE:
                print(f"      máscara: {mask_full['path'].name}")

            # ── tracking da tempestade (só depende da máscara, não de raios) ──
            cur_storm = None
            if mask_full is not None and keo_panels and KEO_MARK_STORMS:
                cur_storm = dict(t0=seg_t0, t1=seg_t1,
                                 clusters=_cluster_extents(mask_full, only_ids=ids_roi))
                storm_history.append(cur_storm)

            plot_map_keogram(
                goes_dt, temp, lat, lon, mask_roi, (seg_t0, seg_t1), x_range, out_path,
                keo_panels=keo_panels, storm_history=storm_history, cur_storm=cur_storm,
            )
            n_png += 1

            cl_txt = f"cl {n_act}/{n_all}" if mask_full is not None else mask_note
            print(f"  {tag} │ {cl_txt:<{w_cl}} │ {time.perf_counter() - t_frame0:.0f} s")

        except Exception as e:
            n_err += 1
            print(f"  {tag} │ ✗ {type(e).__name__}: {str(e)[:60]}")

        finally:
            raw = None   # nada a apagar em disco; garante que os bytes do frame não ficam na memória

    print(f"{'─'*52}")
    print(f"  resumo: {n_frames} frames · {n_png} PNG novos · {n_skip} já existiam · "
          f"{_fmt_dur(time.perf_counter() - t_day0)} · {n_nomask} sem máscara · {n_err} erros")
    print(f"  MAP+KEO → {OUT_DIR / file_date.strftime('%Y%m%d')}")
    if n_frames and n_nomask == n_frames:
        print("  ⚠ nenhuma máscara achada: confira MASK_DIR/MASK_NAME_PATTERNS "
              "(LOG_VERBOSE=True p/ detalhes)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    sess = _build_session()
    dates = _dates_to_process()
    if not dates:
        print("Defina TARGET_DATE ('YYYYMMDD') ou DATE_RANGE (('YYYYMMDD', 'YYYYMMDD')) no bloco CONFIG.")
    for file_date in dates:
        date_str = file_date.strftime("%Y%m%d")
        _process_day(
            file_date,
            frame_time_range=FRAME_TIME_RANGE_BY_DATE.get(date_str, FRAME_TIME_RANGE),
            target_time=TARGET_TIME,
            sess=sess,
        )
    print("\nAll done.")


if __name__ == "__main__":
    main()