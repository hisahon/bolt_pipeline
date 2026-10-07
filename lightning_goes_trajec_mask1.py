"""
plot_lightning_goes.py

Downloads GOES-16/19 channel-13 .nc files from CPTEC, reads them on the fly,
plots the IR brightness temperature with lightning strikes overlaid, then
deletes the .nc file. No .npz conversion.

Duas figuras por frame GOES:
  1) data/images/{YYYYMMDD}/*_CH13.png        mapa IR + raios + IPPs + isoterma
     'lime' (contorno da máscara de clusters);
  2) data/maps/{YYYYMMDD}/LFR/*_LFR.png       mapa com clusters + série temporal
     ACUMULATIVA de LFR (fl/s) por cluster: a cada frame entram só os raios de
     epoch ± LFR_HALF_WINDOW_MIN, e os trechos anteriores são mantidos na figura.
As máscaras vêm dos .npz gravados por detect_clusters_area6 (save_labels_npz).

Folder layout:
    data/
        parquet/    ← resampled lightning .parquet files
        masks/      ← labels de clusters (.npz) gerados por detect_clusters_area6
        goes_tmp/   ← temporary .nc during download (auto-deleted after plot)
        images/     ← output PNG maps (figura 1: mapa IR + raios + IPPs)
        maps/       ← output das figuras de LFR: maps/{YYYYMMDD}/LFR/*_LFR.png

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
import time
import inspect
from pathlib import Path
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
import duckdb
# import netCDF4 as nc4

# módulo compartilhado do projeto (mesmo usado em visualize_hilbert_temporary.py)
from shared.config import dtec_ipp_dir


# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_DIR       = r"data\parquet"
GOES_TMP_DIR      = r"data\goes_tmp"
IMAGE_DIR         = r"data\images"
MAP_DIR           = r"data\maps"      # pasta-base das figuras de LFR: MAP_DIR/{YYYYMMDD}/LFR/
NEW_SAMPLING_RATE = 15
CHANNEL           = 13
GOES_TIMEOUT      = 30.0
VMIN_TEMP         = -80       # °C
VMAX_TEMP         = 40        # °C (a escala realçada vai de -80 a 40 °C)

# Escala realçada do canal 13 (10,35 µm): faixas coloridas de 10 °C entre
# -80 e -30 °C e degradê de cinza claro -> preto de -30 a 40 °C.
# Temperaturas repetidas criam os "saltos" de cor entre as faixas.
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

# ── seleção de data ────────────────────────────────────────────────────────
# TARGET_DATE : processa apenas um dia específico. Formato 'YYYYMMDD' ou None.
# DATE_RANGE  : processa um intervalo (início, fim), ambos 'YYYYMMDD'. Ignorado
#               se TARGET_DATE estiver definido. Use None para desativar.
# Se os dois forem None, processa todos os parquet encontrados (comportamento original).
TARGET_DATE       = "20211023"                       # ex: "20250115"
# TARGET_DATE       = "20260723"                       # ex: "20250115"
TARGET_DATE       = "20231213"                       # ex: "20250115"
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

# ROI_LAT_MIN       = -30
# ROI_LAT_MAX       = -20
# ROI_LON_MIN       = -60
# ROI_LON_MAX       = -50

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
HEXBIN_CMAP       = "inferno"   # escala de cor dos raios (contagem por célula)
HEXBIN_MINCOUNT   = 1         # células com menos raios que isso ficam transparentes
HEXBIN_ALPHA      = 0.85

# Escala de cor fixa (mesmos limites em todos os frames, pra poder comparar
# densidade entre imagens diferentes). Deixe None em qualquer um dos dois
# para que o matplotlib normalize automaticamente por frame (comportamento
# padrão, mas cada PNG fica em uma escala diferente).
HEXBIN_VMIN       = 0        # nº mínimo de raios por célula (cor mais fraca)
HEXBIN_VMAX       = 60       # nº máximo de raios por célula (cor mais forte);
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
IPP_CMAP        = "rainbow"   # escala de cor para o horário (t_hours)
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

# ── máscaras de clusters (saída de detect_clusters_area6 / save_labels_npz) ──
# Formato gravado por save_labels_npz, em <mask_dir>/<nome_do_csv>_labels.npz
# (um arquivo por timestamp do GOES; só existe se houve >=1 cluster aprovado):
#   labels      int16 (ny, nx)   0 = fora; k = cluster k (o mesmo n_clusters do CSV)
#   lats, lons  float32 1D       centros das células da grade (mask_SouthAmerica.npz)
#   time_string str              timestamp da imagem
#   t_threshold, smooth_sigma, min_area_km2, tmin_threshold   (float32)
# Aqui as máscaras servem para:
#   (1) desenhar a ISOTERMA do cluster (contorno da máscara) no mapa; e
#   (2) contar os raios dentro de cluster -> LFR por cluster.
# Os vetores lats/lons para posicionar a isoterma, recortar ao ROI e contar raios
# vêm do PRÓPRIO arquivo de labels; a grid mask (mask_SouthAmerica.npz) só entra
# como reserva se o arquivo de labels não trouxer lats/lons com o shape certo.
GRID_MASK_DIR  = r"D:\goes-data-pipeline\grid_masks"
GRID_MASK_NAME = "mask_SouthAmerica.npz"
GRID_LAT_KEYS  = ("lats", "lat", "latitude", "latitudes")
GRID_LON_KEYS  = ("lons", "lon", "longitude", "longitudes")

# Pasta (busca recursiva) com os *_labels.npz (= MASK_DIR / folders['mask_dir'] da
# detect_clusters_area6; padrão do pipeline: "grid_masks").
MASK_DIR = r"D:\goes-data-pipeline\grid_masks"
MASK_FILE_GLOB = "*_labels.npz"   # padrão de mask_path_from_csv; se nada casar, tenta *.npz
# Como o horário do frame aparece no NOME do arquivo .npz (= folders['time_string']).
# Os padrões são testados em ordem; {Y}{m}{d}{H}{M}{j} = ano, mês, dia, hora, min, DOY.
# Ajuste/adicione o formato que a sua save_labels_npz / mask_path_from_csv usa.
MASK_NAME_PATTERNS = (
    "{Y}{m}{d}{H}{M}",      # 202110231530
    "{Y}{m}{d}_{H}{M}",     # 20211023_1530
    "{Y}{m}{d}T{H}{M}",     # 20211023T1530
    "{Y}{j}{H}{M}",         # 2021296 1530 (ano + DOY)
)
# Nomes aceitos para a chave da matriz de labels dentro do .npz (1ª que existir)
MASK_LABELS_KEYS = ("labels", "labels_filtered", "mask")

# ── figura 1: mapa (como antes) + isoterma da máscara em 'lime' ─────────────
PLOT_MAP_FIGURE    = True
PLOT_ISOTHERM      = True
ISOTHERM_COLOR     = "lime"
ISOTHERM_LINEWIDTH = 1.4

# ── figura 2: LFR (lightning flash rate, flashes/s) por cluster ─────────────
# Painel direito = SÉRIE TEMPORAL do LFR em fl/s, construída de forma ACUMULATIVA:
#   LFR(t) = nº de raios dentro da máscara do cluster no bin de tempo / largura do bin (s).
# A cada frame GOES lido entram SÓ os raios da janela [epoch - LFR_HALF_WINDOW_MIN,
# epoch + LFR_HALF_WINDOW_MIN) -- meia-aberta, para que o raio exatamente na borda
# entre dois frames vizinhos não seja contado duas vezes --, contados com a máscara
# DAQUELE frame, em bins de LFR_BIN_SECONDS. O trecho calculado é guardado
# (lfr_history) e redesenhado nas figuras seguintes: a série vai "crescendo" frame
# a frame. Só os clusters do frame ATUAL são coloridos (mesmas cores do mapa) e
# aparecem na legenda; os clusters dos frames anteriores ficam numa cor única, só como
# contexto, porque o id do cluster é atribuído frame a frame (a mesma cor em frames
# diferentes não significaria o mesmo sistema). A curva Total (soma dos clusters
# de cada frame) não depende dos ids e é a curva contínua da série: linha preta
# grossa, na camada de baixo, com área sombreada sob ela.
# A janela do frame atual aparece sombreada e o seu epoch com uma linha pontilhada.
# Os raios são contados na máscara COMPLETA (sem recorte do ROI), então um cluster
# que sai do ROI não perde raios; só entram na figura os clusters que tocam o ROI.
PLOT_LFR_FIGURE    = True
LFR_DIR_NAME       = "LFR"        # subpasta de MAP_DIR/{YYYYMMDD}/ onde saem as figuras de LFR
LFR_BIN_SECONDS    = NEW_SAMPLING_RATE   # largura do bin (s); 15 s = resolução do parquet.
                                         # Bins maiores (30, 60 s) suavizam a série.
LFR_HALF_WINDOW_MIN = 5           # meia-largura da janela de raios de cada frame (min): epoch ± 5.
                                  # None = metade do passo entre frames (5 min p/ passo de 10 min).
LFR_FIXED_XAXIS    = True         # True = eixo x fixo (1º frame − janela → último frame + janela): a série
                                  # "enche" da esquerda p/ a direita; False = eixo cresce junto com os dados
LFR_YMAX           = 20           # eixo y fixo (fl/s) p/ comparar frames; None = automático
LFR_STRIKE_SIZE    = 8
LFR_ID_LABEL_DLAT  = 1.0          # deslocamento (graus de latitude, p/ cima) dos números dos clusters no mapa
# Camadas do painel (de baixo p/ cima): janela do frame < área do Total < linha do Total
# (GROSSA) < clusters de frames anteriores (cor única) < clusters do frame atual (coloridos).
LFR_HISTORY_COLOR     = "#731dd8" # clusters de frames anteriores: UMA cor só (o id muda de frame p/ frame), também na frente.
                                  # Fora das cores dos primeiros clusters (azul, laranja...), que seriam confundidas
                                  # com os clusters 1/2 do frame atual
LFR_HISTORY_LINEWIDTH = 1.2       # espessura das linhas de cluster do histórico
LFR_CLUSTER_LINEWIDTH = LFR_HISTORY_LINEWIDTH   # clusters do frame ATUAL (coloridos, como no mapa): MESMA espessura
                                                # do histórico (para usar valores diferentes, troque por um número)
LFR_HISTORY_ALPHA     = 1.
LFR_TOTAL_LINEWIDTH   = 2.0       # linha Total (preta): bem mais grossa que as dos clusters e na camada de baixo, então
                                  # sobra uma borda preta visível dos dois lados mesmo quando o Total = 1 cluster
LFR_TOTAL_FILL_COLOR  = "0.5"     # área sombreada sob a curva Total
LFR_TOTAL_FILL_ALPHA  = 0.3       # 0 = sem área sombreada

# ── log no console ──────────────────────────────────────────────────────────
# False = cabeçalho compacto + UMA linha por frame + resumo final.
# True  = também mostra detalhes (nomes de máscara de exemplo, chaves do .npz,
#         resolução do GOES, cada PNG salvo).
LOG_VERBOSE        = False
# ─────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("OMP_NUM_THREADS", "1")

PARQUET_DIR  = Path(PARQUET_DIR)
GOES_TMP_DIR = Path(GOES_TMP_DIR)
IMAGE_DIR    = Path(IMAGE_DIR)
MAP_DIR      = Path(MAP_DIR)
MASK_DIR     = Path(MASK_DIR)
GRID_MASK_DIR = Path(GRID_MASK_DIR)





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
        # (2026, 204): {
        #     "tbeg": 16, 
        #     "tend": 17 + 40/60,
        #     "pairs": [
        #         "scaq · R18", "stha · R18", "stha · G29", 
        #         "itai · R18", "itai · G29", "prcl · G29", 
        #         "puma · G11", "fosa · R18", "rosa · G29"
        #     ]
        # },
        (2026, 203): {
            "tbeg": 15, 
            "tend": 19,
            "pairs": [
                "fosa · G11", "msdr · R24", "ppte · E23", 
                "ppte · G12", "ppte · G13", "prma · G12", 
                "prma · E23", "puma · R24", "py01 · G11"
            ]
        },
        (2026,204):{
            "tbeg": 16,
            "tend": 21,
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
            "tend": 17 + 30/60,
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
    if resolution_key != _last_printed_resolution and LOG_VERBOSE:
        print(f"   Resolução nativa GOES detectada: ~{lat_res:.4f}° lat × {lon_res:.4f}° lon "
              f"(fator {HEXBIN_RESOLUTION_FACTOR}) → hexbin gridsize=({nx}, {ny})")
        _last_printed_resolution = resolution_key

    return (nx, ny)



# ── máscaras de clusters (.npz salvos por detect_clusters_area6) ─────────────
_MASK_INDEX = None       # cache: [(Path, só-dígitos do caminho relativo)] de MASK_DIR
_MASK_TS_INDEX = None    # cache: [(Path, dígitos do time_string, time_string como datetime)] (fallback)
_MASK_MISS_LOGGED = 0    # quantos diagnósticos de "não achei" já foram impressos


def _digits(s) -> str:
    """Só os dígitos: '2021-10-23 15:30' / '20211023_1530' / '202110231530' -> '202110231530'."""
    return re.sub(r"\D", "", str(s))


def _mask_time_strings(goes_dt: datetime) -> list:
    """
    Chaves só-dígitos do horário do frame (ex.: '202110231530' e '20212961530').
    A comparação ignora separadores (_ - : T) tanto no nome/caminho do arquivo
    quanto no 'time_string' interno, então o formato exato deixa de importar.
    """
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
    """
    Interpreta o time_string gravado na máscara como data/hora (minuto):
    AAAAMMDDHHMM com ou sem separadores ('2021-10-23 15:30', '20211023_1530', ...)
    ou formatos com o dia primeiro / hora primeiro ('15:30 23/10/2021').
    """
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
    """
    Varre MASK_DIR uma única vez e monta o índice dos arquivos de labels.
    Devolve quantos arquivos foram indexados (0 se a pasta não existe/está vazia).
    Detalhes (exemplos de nome, chaves do .npz) só aparecem com LOG_VERBOSE.
    """
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
    Acha o .npz de labels do frame, comparando só os DÍGITOS do horário:
    1) no caminho relativo do arquivo (pasta por data + nome);
    2) se nada casar, no campo 'time_string' gravado DENTRO dos .npz
       (índice montado uma única vez, só na primeira falha).
    """
    global _MASK_TS_INDEX, _MASK_MISS_LOGGED
    _ensure_mask_index()

    wanted = _mask_time_strings(goes_dt)
    for key in wanted:
        for p, dig in _MASK_INDEX:
            if key in dig:
                return p

    # fallback: time_string dentro do arquivo (dígitos OU interpretado como data/hora)
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
    """
    Procura os vetores lat/lon num .npz aberto (aceita apelidos de chave) e
    devolve (lats, lons) 1D, ou (None, None). Se vierem como malha 2D
    (meshgrid), reduz para 1D.
    """
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
    """
    Lê (uma vez) os vetores lat/lon da grid mask -- a mesma mask_SouthAmerica.npz
    que detect_clusters_area6 usa. Procura GRID_MASK_NAME em GRID_MASK_DIR; se não
    achar, qualquer '*SouthAmerica*.npz' por lá; por último, no diretório atual.
    """
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
    Carrega as labels do frame. Devolve dict(labels, lats, lons, t_threshold)
    ou None se o frame não tem máscara (nenhum cluster aprovado nesse horário
    -- detect_clusters_area6 só grava o arquivo quando há >=1 cluster).
    """
    path = _find_mask_file(goes_dt)
    if path is None:
        return None

    with np.load(path, allow_pickle=True) as d:
        key = next((k for k in MASK_LABELS_KEYS if k in d.files), None)
        if key is None:
            # fallback: 1º array 2D inteiro/booleano do arquivo
            key = next((k for k in d.files
                        if d[k].ndim == 2 and (np.issubdtype(d[k].dtype, np.integer)
                                               or d[k].dtype == bool)), None)
            if key is None:
                raise KeyError(f"{path.name}: sem matriz de labels; chaves = {d.files}")
            print(f"   [WARN] chave de labels não reconhecida em {path.name}; usando '{key}' "
                  f"(chaves: {d.files}). Ajuste MASK_LABELS_KEYS se estiver errado.")
        labels = np.asarray(d[key]).astype(np.int32)

        # vetores lat/lon: os gravados no PRÓPRIO arquivo de labels (centros das células
        # da grade usada na detecção); se faltarem ou o shape não casar, usa a grid mask
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
    """Recorta as labels para o ROI (mesmo critério de _crop_to_roi)."""
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


def _nearest_index(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Índice do pixel mais próximo de cada valor numa grade 1D (crescente ou
    decrescente). -1 para valores fora da grade (> meio pixel além da borda).
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    order  = np.argsort(coords)
    sc     = coords[order]
    half   = np.median(np.diff(sc)) / 2.0 if len(sc) > 1 else 0.0

    pos   = np.clip(np.searchsorted(sc, values), 1, len(sc) - 1)
    left  = sc[pos - 1]
    right = sc[pos]
    pos   = np.where((values - left) <= (right - values), pos - 1, pos)

    idx = order[pos]
    out_of_grid = (values < sc[0] - half) | (values > sc[-1] + half) | ~np.isfinite(values)
    return np.where(out_of_grid, -1, idx)


def _strike_cluster_ids(labels, lats, lons, strikes: pd.DataFrame) -> np.ndarray:
    """Id do cluster de cada raio (0 = fora de qualquer cluster / fora da grade)."""
    if strikes.empty:
        return np.zeros(0, dtype=np.int32)
    iy = _nearest_index(lats, strikes["latitude"].to_numpy())
    ix = _nearest_index(lons, strikes["longitude"].to_numpy())
    ok = (iy >= 0) & (ix >= 0)
    out = np.zeros(len(strikes), dtype=np.int32)
    out[ok] = labels[iy[ok], ix[ok]]
    return out


def compute_lfr_per_cluster(mask: dict, strikes: pd.DataFrame, window_min: float) -> pd.DataFrame:
    """
    LFR por cluster = (raios dentro do cluster na janela) / janela [min].
    Colunas: cluster, n_flashes, lfr (raios/min).
    """
    labels = mask["labels"]
    ids = np.unique(labels[labels > 0])
    if ids.size == 0:
        return pd.DataFrame(columns=["cluster", "n_flashes", "lfr"])

    sid = _strike_cluster_ids(labels, mask["lats"], mask["lons"], strikes)
    counts = np.bincount(sid[sid > 0], minlength=int(labels.max()) + 1)
    return pd.DataFrame({
        "cluster":   ids,
        "n_flashes": counts[ids],
        "lfr":       counts[ids] / float(window_min),
    })


def compute_lfr_segment(mask: dict, strikes: pd.DataFrame, t_start, t_end,
                        bin_s: float, only_ids=None) -> dict:
    """
    LFR (flashes/s) por cluster num ÚNICO trecho de tempo [t_start, t_end): a
    janela do frame GOES (epoch ± LFR_HALF_WINDOW_MIN), com a máscara desse frame.

    O trecho é dividido em bins de largura ~bin_s; em cada bin,
    LFR = nº de raios dentro do cluster / largura do bin [s]. A janela é
    meia-aberta (raio em t_end não entra), de modo que o raio exatamente na
    borda entre dois frames consecutivos conta só no frame seguinte.

    Devolve dict(
        t_start, t_end : Timestamps do trecho,
        edges    : DatetimeIndex (n+1) com as bordas dos bins,
        width_s  : largura efetiva do bin (s),
        rates    : {id_cluster: array (n,) em fl/s},
        counts   : {id_cluster: nº de raios no trecho},
        mean_fls : {id_cluster: LFR médio no trecho (fl/s)},
        total    : array (n,) com a soma dos clusters, ou None se não houver cluster,
    )
    """
    t_start, t_end = pd.Timestamp(t_start), pd.Timestamp(t_end)
    dur_s = (t_end - t_start).total_seconds()
    nbins = max(1, int(round(dur_s / float(bin_s))))
    edges_s = np.linspace(0.0, dur_s, nbins + 1)
    width_s = dur_s / nbins
    edges = t_start + pd.to_timedelta(edges_s, unit="s")

    labels = mask["labels"]
    ids = np.unique(labels[labels > 0])
    if only_ids is not None:
        ids = np.intersect1d(ids, np.asarray(only_ids))

    if len(strikes):
        t_sec = (pd.to_datetime(strikes["time_stamp"]) - t_start).dt.total_seconds().to_numpy()
        keep = (t_sec >= 0.0) & (t_sec < dur_s)          # janela meia-aberta [t_start, t_end)
        strikes, t_sec = strikes[keep], t_sec[keep]
        sid = _strike_cluster_ids(labels, mask["lats"], mask["lons"], strikes)
    else:
        sid = np.zeros(0, dtype=np.int32)
        t_sec = np.zeros(0)

    rates, counts, mean_fls = {}, {}, {}
    for cid in ids:
        sel = sid == cid
        h, _ = np.histogram(t_sec[sel], bins=edges_s)
        rates[int(cid)] = h / width_s
        counts[int(cid)] = int(np.count_nonzero(sel))
        mean_fls[int(cid)] = counts[int(cid)] / max(dur_s, 1e-9)

    total = np.sum(list(rates.values()), axis=0) if rates else None

    return dict(t_start=t_start, t_end=t_end, edges=edges, width_s=width_s,
                rates=rates, counts=counts, mean_fls=mean_fls, total=total)


# ── feições geográficas + grade (compartilhadas pelas duas figuras) ─────────
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
    """
    Contorno 'lime' da máscara de clusters (isoterma de t_threshold, já suavizada
    pelo detect_clusters_area6). Devolve True se desenhou algo.
    """
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


# ── map plot (cartopy) ──────────────────────────────────────────────────────
def plot_goes_lightning(goes_dt, temp, lat, lon, lightning, out_path, ipp=None, ipp_epoch=None,
                        mask=None):
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
    _add_map_features(ax)

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

    # isoterma lime da máscara de clusters (zorder 5: acima da trajetória de IPPs,
    # abaixo do marcador de epoch)
    has_iso = False
    if mask is not None:
        has_iso = _draw_isotherm(ax, mask, proj)

    if has_epoch:
        ax.scatter(
            ipp_epoch[IPP_LON_COL], ipp_epoch[IPP_LAT_COL],
            marker=IPP_EPOCH_MARKER_STYLE, s=IPP_EPOCH_MARKER_SIZE,
            facecolors=IPP_EPOCH_COLOR, edgecolors=IPP_EPOCH_EDGE_COLOR,
            linewidths=IPP_EPOCH_EDGE_WIDTH, transform=proj, zorder=6,
            label="IPP @ epoch",
        )

    if has_epoch or has_iso:
        ax.legend(loc="lower left", fontsize=12, framealpha=0.8)

    _add_gridlines(ax)

    ax.set_title(
        rf"\textbf{{{goes_dt.strftime('%H:%M')} UTC - {goes_dt.strftime('%Y/%m/%d')} GOES Infrared (10.3 $\mu$m)}}"
    )


    # add_provenance_footer(fig, storage_path=str(IMAGE_DIR) + os.sep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    if LOG_VERBOSE:
        print(f"   Saved → {out_path.name}")
    return out_path


# ── figura 2: LFR por cluster ───────────────────────────────────────────────
def _draw_lfr_panel(axb, lfr_history, cur_seg, goes_dt, frame_window, x_range, color_of):
    """
    Painel direito ACUMULATIVO: desenha todos os trechos de LFR já calculados
    (lfr_history = lista de saídas de compute_lfr_segment, em ordem cronológica,
    incluindo o do frame atual = cur_seg) como degraus em fl/s.
      - clusters do frame ATUAL: coloridos (cor do id, a mesma do mapa) e na legenda;
      - clusters de frames anteriores: cor única fina (LFR_HISTORY_COLOR), só contexto (o id muda entre
        frames, então a cor não significaria o mesmo sistema), sem legenda;
      - Total (soma dos clusters de cada frame): linha preta grossa na camada de
        baixo, com área sombreada sob ela, por trás de TODAS as linhas de cluster
        (histórico e frame atual, mais finas). Sempre desenhado, também com 1 cluster:
        como fica atrás e é mais grosso, aparece como uma borda ao redor da linha.
    cur_seg = None quando o frame não tem máscara (só o histórico é mostrado).
    """
    axb.set_ylabel(r"Lightning flash rate (fl s$^{-1}$)")
    axb.grid(True, linestyle="--", alpha=0.4)
    axb.set_axisbelow(True)

    f0, f1 = (pd.Timestamp(t).to_pydatetime() for t in frame_window)
    axb.axvspan(f0, f1, color="0.85", alpha=0.6, zorder=1)   # janela do frame atual (fora da legenda)

    ymax = 0.0
    for seg in lfr_history:
        is_cur = seg is cur_seg
        x = seg["edges"].to_pydatetime()
        for cid in sorted(seg["rates"]):
            r = seg["rates"][cid]
            ymax = max(ymax, float(r.max()))
            if is_cur:   # frame atual: colorido, com legenda; zorder 5 = camada de cima
                axb.step(
                    x, np.append(r, r[-1]), where="post", color=color_of(cid),
                    linewidth=LFR_CLUSTER_LINEWIDTH, zorder=5,
                    # label=rf"Cluster {cid} (mean {seg['mean_fls'][cid]:.3f} fl s$^{{-1}}$)",
                    label=rf"Cluster {cid}",
                )
            else:        # frames anteriores: cor única fina, sem legenda (zorder 4, ainda acima do Total)
                axb.step(
                    x, np.append(r, r[-1]), where="post", color=LFR_HISTORY_COLOR,
                    linewidth=LFR_HISTORY_LINEWIDTH, alpha=LFR_HISTORY_ALPHA, zorder=4,
                )
        # Total: sempre desenhado (também com 1 cluster), na camada de baixo -- área (zorder 1.5)
        # e linha grossa (zorder 3), ambas atrás das linhas de cluster (4 e 5)
        if seg["total"] is not None:
            tot = seg["total"]
            y_tot = np.append(tot, tot[-1])
            ymax = max(ymax, float(tot.max()))
            if LFR_TOTAL_FILL_ALPHA > 0:
                axb.fill_between(x, 0, y_tot, step="post", color=LFR_TOTAL_FILL_COLOR,
                                 alpha=LFR_TOTAL_FILL_ALPHA, linewidth=0, zorder=1.5)
            axb.step(x, y_tot, where="post", color="black", linestyle="-",
                     linewidth=LFR_TOTAL_LINEWIDTH, zorder=3,
                     label="Total" if is_cur else None)

    axb.axvline(goes_dt, color="black", linestyle=":", linewidth=1.2, zorder=2)

    if LFR_FIXED_XAXIS and x_range is not None:
        x0, x1 = (pd.Timestamp(t).to_pydatetime() for t in x_range)
    else:   # eixo cresce junto com os dados
        x0 = min([f0] + [s["t_start"].to_pydatetime() for s in lfr_history])
        x1 = f1
    axb.set_xlim(x0, x1)
    axb.set_ylim(0, LFR_YMAX if LFR_YMAX is not None else max(ymax * 1.5, 1.5 / LFR_BIN_SECONDS))
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axb.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    axb.set_xlabel("Time (UTC)")

    handles, _ = axb.get_legend_handles_labels()
    if handles:
        axb.legend(loc="upper right", fontsize=11 if len(handles) <= 6 else 9, framealpha=0.85,
                   ncol=1 if len(handles) <= 6 else 2)
    else:
        note = ("No cluster mask for this frame" if cur_seg is None
                else "No cluster in the ROI for this frame")
        axb.text(0.5, 0.97, note, ha="center", va="top", transform=axb.transAxes,
                 fontsize=14, zorder=8,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9))

    half_min = (f1 - f0).total_seconds() / 120.0
    axb.set_title(
        rf"\textbf{{LFR per cluster}} ({LFR_BIN_SECONDS:g}-s bins, epoch $\pm${half_min:g} min)",
        fontsize=14,
    )


def plot_lfr_clusters(goes_dt, temp, lat, lon, strikes_roi, mask_roi, lfr_history, cur_seg,
                      frame_window, x_range, out_path):
    """
    Esquerda: mapa GOES + isoterma lime + id de cada cluster + raios (coloridos
    pelo cluster em que caem; brancos = fora de cluster ou frame sem máscara).
    Direita: série temporal ACUMULATIVA do LFR (flashes/s) por cluster, na mesma
    cor do mapa. lfr_history = trechos já calculados (um por frame, saída de
    compute_lfr_segment); cur_seg = trecho do frame atual (None se não há máscara);
    frame_window = (t0, t1) do frame atual; x_range = (início, fim) do eixo x fixo.
    A legenda mostra só os clusters do frame atual.
    Se não houver máscara (nenhum cluster no frame), sai a figura sem clusters,
    para manter a sequência de frames completa em animações.
    """
    if CROP_TO_ROI:
        lon_min, lon_max = ROI_LON_MIN, ROI_LON_MAX
        lat_min, lat_max = ROI_LAT_MIN, ROI_LAT_MAX
    else:
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 8))

    # mapa quadrado (6,4 x 6,4 pol) -- mesma lógica de proporção da figura 1
    ax       = fig.add_axes([0.06, 0.12, 0.40, 0.80], projection=proj)
    cax_temp = fig.add_axes([0.475, 0.12, 0.012, 0.80])
    axb      = fig.add_axes([0.62, 0.12, 0.36, 0.80])

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    im = ax.imshow(
        temp, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max],
        transform=proj, cmap=CMAP_GOES, vmin=VMIN_TEMP, vmax=VMAX_TEMP,
        interpolation="nearest", zorder=1,
    )
    cbar = fig.colorbar(im, cax=cax_temp)
    cbar.set_label("Brightness Temperature (°C)")
    _add_map_features(ax)

    cmap_id = plt.get_cmap("tab10")
    color_of = lambda cid: cmap_id((int(cid) - 1) % 10)

    has_mask = mask_roi is not None and (mask_roi["labels"] > 0).any()
    labels = mask_roi["labels"] if has_mask else None

    # raios: SEMPRE desenhados (branco = fora de cluster / sem máscara;
    # colorido = dentro de um cluster)
    if len(strikes_roi):
        if has_mask:
            sid = _strike_cluster_ids(labels, mask_roi["lats"], mask_roi["lons"], strikes_roi)
        else:
            sid = np.zeros(len(strikes_roi), dtype=np.int32)
        s_lon = strikes_roi["longitude"].to_numpy()
        s_lat = strikes_roi["latitude"].to_numpy()
        out = sid == 0
        ax.scatter(
            s_lon[out], s_lat[out],
            s=LFR_STRIKE_SIZE, facecolors="white", edgecolors="black", alpha=0.85,
            linewidths=0.3, transform=proj, zorder=3,
        )
        inn = ~out
        if inn.any():
            ax.scatter(
                s_lon[inn], s_lat[inn],
                s=LFR_STRIKE_SIZE * 1.6, c=[color_of(c) for c in sid[inn]],
                edgecolors="black", linewidths=0.3, transform=proj, zorder=4,
            )

    if has_mask:
        _draw_isotherm(ax, mask_roi, proj)

        # id do cluster no centróide (da parte do cluster que está dentro do ROI), deslocado
        # LFR_ID_LABEL_DLAT graus p/ cima (não cobre a região do cluster); se o deslocamento
        # levaria o marcador pra fora do mapa, ele é limitado a 0,5° abaixo da borda superior
        for cid in np.unique(labels[labels > 0]):
            rows, cols = np.nonzero(labels == cid)
            ax.text(
                float(mask_roi["lons"][cols].mean()),
                min(float(mask_roi["lats"][rows].mean()) + LFR_ID_LABEL_DLAT, lat_max - 0.5),
                _bold(int(cid)), color="white", fontsize=12,
                ha="center", va="center", transform=proj, zorder=8,
                bbox=dict(boxstyle="circle,pad=0.12", fc=color_of(cid), ec="black", lw=0.8),
            )
        ax.legend(loc="lower left", fontsize=12, framealpha=0.8)

    _add_gridlines(ax)
    ax.set_title(
        rf"\textbf{{{goes_dt.strftime('%H:%M')} UTC - {goes_dt.strftime('%Y/%m/%d')} GOES Infrared (10.3 $\mu$m)}}"
    )

    # ── série temporal ACUMULATIVA do LFR (flashes/s) ───────────────────────
    _draw_lfr_panel(axb, lfr_history, cur_seg, goes_dt, frame_window, x_range, color_of)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    if LOG_VERBOSE:
        print(f"   Saved → {out_path.parent.name}/{out_path.name}")
    return out_path


# ── seleção de data: filtro de parquet ────────────────────────────────────────
def _extract_date(p: Path):
    m = re.search(r"(\d{8})", p.name)
    return datetime.strptime(m.group(1), "%Y%m%d").date() if m else None


def _filter_by_date(files: list) -> list:
    if TARGET_DATE:
        target = datetime.strptime(TARGET_DATE, "%Y%m%d").date()
        filtered = [p for p in files if _extract_date(p) == target]
        print(f"Filtro {TARGET_DATE}: {len(filtered)} parquet(s)")
        return filtered

    if DATE_RANGE:
        start = datetime.strptime(DATE_RANGE[0], "%Y%m%d").date()
        end   = datetime.strptime(DATE_RANGE[1], "%Y%m%d").date()
        filtered = [p for p in files if _extract_date(p) and start <= _extract_date(p) <= end]
        print(f"Filtro {DATE_RANGE[0]}–{DATE_RANGE[1]}: {len(filtered)} parquet(s)")
        return filtered

    return files


# ── main ──────────────────────────────────────────────────────────────────────
def _fmt_dur(seconds: float) -> str:
    """45.2 -> '45 s'; 133 -> '2 min 13 s'; 3725 -> '1 h 02 min'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def _process_day(pq_path, file_date, ipp_time_range, ipp_pairs, target_time, sess, frame_time_range=None):
    t_day0 = time.perf_counter()
    doy_day = file_date.timetuple().tm_yday
    print(f"\n{'═'*52}")

    lightning = pd.read_parquet(pq_path)
    lightning["time_stamp"] = pd.to_datetime(lightning["time_stamp"])
    print(f"{file_date} (DOY {doy_day:03d}) · {len(lightning):,} raios")
    win_txt = tt_txt = ipp_txt = ""

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

    # ── carrega a trajetória de IPPs do dia (uma vez, reaproveitada em todos os frames) ──
    ipp_day = pd.DataFrame()
    if PLOT_IPP_TRAJECTORY:
        doy_of_year = file_date.timetuple().tm_yday
        ipp_day = load_ipp_trajectory(
            year=file_date.year, doy=doy_of_year,
            lat_range=IPP_LAT_RANGE, lon_range=IPP_LON_RANGE,
            time_range=ipp_time_range, stations=IPP_STATIONS, pairs=ipp_pairs,
        )
        ipp_txt = f"{len(ipp_day):,} IPPs"

    # infer CPTEC sampling rate from the gaps between consecutive frames
    frame_dts = sorted([_dt_from_name(n) for n in day_names if _dt_from_name(n)])
    if len(frame_dts) >= 2:
        gaps = [(frame_dts[i+1] - frame_dts[i]).seconds // 60
                for i in range(len(frame_dts) - 1)]
        cptec_sampling_min = min(gaps)
    else:
        cptec_sampling_min = 10   # fallback if only one frame
    n_frames = len(day_names)
    n_mask_files = _ensure_mask_index() if (PLOT_ISOTHERM or PLOT_LFR_FIGURE) else 0
    partes = [f"{n_frames} frames" + (f" {win_txt}" if win_txt else ""),
              f"passo {cptec_sampling_min} min"]
    if tt_txt:
        partes.append(tt_txt)
    if ipp_txt:
        partes.append(ipp_txt)
    if PLOT_ISOTHERM or PLOT_LFR_FIGURE:
        partes.append(f"{n_mask_files} máscaras")
    print("  " + " · ".join(partes))
    print(f"{'─'*52}")
    w_tag, w_cl, w_st, w_lfr = len(f"[{n_frames}/{n_frames}] 00:00"), 7, 18, 12

    n_png = n_nomask = n_err = 0

    # ── série de LFR ACUMULATIVA ──
    # Cada frame contribui só com os raios de [epoch - lfr_half, epoch + lfr_half);
    # o trecho calculado vai para lfr_history e é redesenhado nas figuras seguintes
    # (o histórico é zerado a cada dia/run, pois _process_day é chamada uma vez por dia).
    lfr_half = timedelta(minutes=(LFR_HALF_WINDOW_MIN if LFR_HALF_WINDOW_MIN is not None
                                  else cptec_sampling_min / 2))
    lfr_history = []
    x_range = None
    if PLOT_LFR_FIGURE and frame_dts:
        # eixo x fixo: do começo da janela do 1º frame ao fim da janela do último
        x_range = (pd.Timestamp(frame_dts[0] - lfr_half), pd.Timestamp(frame_dts[-1] + lfr_half))
        if LOG_VERBOSE:
            print(f"  LFR: epoch ± {lfr_half.total_seconds() / 60:g} min · bins {LFR_BIN_SECONDS:g} s · "
                  f"eixo x {x_range[0]:%H:%M}–{x_range[1]:%H:%M}")

    # ordem cronológica (a série acumula frame a frame, então a ordem importa)
    for i_frame, name in enumerate(
            sorted(day_names, key=lambda n: _dt_from_name(n) or datetime.min), 1):
        goes_dt = _dt_from_name(name)
        if not goes_dt:
            continue

        t_frame0 = time.perf_counter()
        tag = f"[{i_frame:>{len(str(n_frames))}}/{n_frames}] {goes_dt:%H:%M}"

        nc_path = _download_nc(sess, urljoin(url, name), name)
        if not nc_path:
            n_err += 1
            print(f"  {tag} │ ✗ download")
            continue

        try:
            temp, lat, lon = _read_nc(nc_path)
            temp, lat, lon = _crop_to_roi(temp, lat, lon)

            half_window = timedelta(minutes=cptec_sampling_min / 2)
            t_min  = pd.Timestamp(goes_dt - half_window)
            t_max  = pd.Timestamp(goes_dt + half_window)
            nearby_all = lightning[
                (lightning["time_stamp"] >= t_min) &
                (lightning["time_stamp"] <= t_max)
            ]   # janela do frame, SEM recorte de ROI (usado na contagem de LFR)
            nearby = nearby_all

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

            # ── máscara de clusters do frame (None = nenhum cluster aprovado) ──
            mask_full, mask_roi = None, None
            n_all = n_roi = 0
            mask_note = "—"
            if PLOT_ISOTHERM or PLOT_LFR_FIGURE:
                try:
                    mask_full = _load_cluster_mask(goes_dt)
                    mask_note = "sem máscara"
                except Exception as e:   # máscara ilegível não pode derrubar o mapa
                    mask_note = f"⚠ máscara: {type(e).__name__}: {str(e)[:40]}"
                    mask_full = None
                if mask_full is None:
                    n_nomask += 1
                else:
                    mask_roi = _crop_mask_to_roi(mask_full)
                    n_all = len(np.unique(mask_full["labels"][mask_full["labels"] > 0]))
                    n_roi = len(np.unique(mask_roi["labels"][mask_roi["labels"] > 0]))
                    if LOG_VERBOSE:
                        print(f"      máscara: {mask_full['path'].name}")

            ipp_epoch = pd.DataFrame()
            if PLOT_IPP_TRAJECTORY and IPP_EPOCH_MARKER:
                ipp_epoch = _nearest_ipp_epoch(
                    ipp_day, goes_dt, max_gap_min=IPP_EPOCH_MAX_GAP_MIN
                )

            png_frame = 0
            lfr_txt = "—"

            # figura 1: mapa como antes + isoterma lime
            if PLOT_MAP_FIGURE:
                plot_goes_lightning(
                    goes_dt, temp, lat, lon, nearby, out_path,
                    ipp=ipp_day, ipp_epoch=ipp_epoch,
                    mask=mask_roi if PLOT_ISOTHERM else None,
                )
                png_frame += 1

            # figura 2: LFR por cluster (série acumulativa)
            if PLOT_LFR_FIGURE:
                seg_t0 = pd.Timestamp(goes_dt - lfr_half)
                seg_t1 = pd.Timestamp(goes_dt + lfr_half)
                cur_seg = None
                if mask_full is not None:
                    # só os raios de epoch ± lfr_half, SEM recorte de ROI
                    strikes_seg = lightning[
                        (lightning["time_stamp"] >= seg_t0) &
                        (lightning["time_stamp"] < seg_t1)
                    ]
                    # só os clusters que tocam o ROI entram na figura
                    ids_roi = np.unique(mask_roi["labels"][mask_roi["labels"] > 0])
                    cur_seg = compute_lfr_segment(
                        mask_full, strikes_seg, seg_t0, seg_t1, LFR_BIN_SECONDS,
                        only_ids=ids_roi,
                    )
                    lfr_history.append(cur_seg)   # guardado p/ as próximas figuras
                    vals = sorted(cur_seg["mean_fls"].items(), key=lambda kv: -kv[1])
                    if vals:
                        lfr_txt = f"LFR {sum(v for _, v in vals):.3f}"
                        if LOG_VERBOSE:   # detalhe por cluster (3 maiores) só no modo verboso
                            top = " · ".join(f"#{c} {v:.3f}" for c, v in vals[:3])
                            extra = f" (+{len(vals) - 3})" if len(vals) > 3 else ""
                            lfr_txt += f" ({top}{extra})"
                    else:
                        lfr_txt = "sem cluster no ROI"
                else:
                    lfr_txt = mask_note

                lfr_path = (MAP_DIR / file_date.strftime('%Y%m%d') / LFR_DIR_NAME
                            / out_name.replace(".png", "_LFR.png"))
                plot_lfr_clusters(
                    goes_dt, temp, lat, lon, nearby, mask_roi, lfr_history, cur_seg,
                    (seg_t0, seg_t1), x_range, lfr_path,
                )
                png_frame += 1

            n_png += png_frame
            cl_txt = f"cl {n_all}/{n_roi}" if mask_full is not None else "cl —"
            st_txt = (f"raios {len(nearby_all):,}/{len(nearby):,}" if CROP_TO_ROI
                      else f"raios {len(nearby_all):,}")
            print(f"  {tag} │ {cl_txt:<{w_cl}} │ {st_txt:<{w_st}} │ {lfr_txt:<{w_lfr}} │ "
                  f"{time.perf_counter() - t_frame0:.0f} s")

        except Exception as e:
            n_err += 1
            print(f"  {tag} │ ✗ {type(e).__name__}: {str(e)[:60]}")

        finally:
            try:
                if nc_path.exists():
                    nc_path.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"{'─'*52}")
    print(f"  resumo: {n_frames} frames · {n_png} PNG · {_fmt_dur(time.perf_counter() - t_day0)} · "
          f"{n_nomask} sem máscara · {n_err} erros")
    print(f"  CH13 → {IMAGE_DIR / file_date.strftime('%Y%m%d')}")
    if PLOT_LFR_FIGURE:
        print(f"  LFR  → {MAP_DIR / file_date.strftime('%Y%m%d') / LFR_DIR_NAME}")
    if (PLOT_ISOTHERM or PLOT_LFR_FIGURE) and n_frames and n_nomask == n_frames:
        print("  ⚠ nenhuma máscara achada: confira MASK_DIR/MASK_NAME_PATTERNS "
              "(LOG_VERBOSE=True p/ detalhes)")


sess = _build_session()

if manual_runs:
    print(f"manual_runs: {len(manual_runs)} run(s)")
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