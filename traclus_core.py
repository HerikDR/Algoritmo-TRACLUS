"""
traclus_core.py
Implementação do núcleo do pipeline TRACLUS
─────────────────────────────────────────────────────────────────────
"""
# Bibliotecas essenciais
import json
from collections import deque

import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

#-Constantes-Globais─────────────────────────────────────────────────

RAIO_TERRA_KM = 6371.0 # Raio da terra para projeção equirretangular
EPSILON = 1e-6         # Piso numério para evitar log2(o) e divisão por 0
NAO_CLASSIFICADO = -2  # Rótulo para segmentos não visitados
RUIDO = -1             # Rótulo para segmentos de ruído

#-Funções-de-Projeção-Geográfica─────────────────────────────────────

def latlon_para_xy(lat, lon, lat_ref, lon_ref, R=RAIO_TERRA_KM):
    # Projeta lat/lon em coordenadas cartesianas com cosseno
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)
    x = R * (lon_rad - lon_ref_rad) * np.cos(lat_ref_rad)
    y = R * (lat_rad - lat_ref_rad)
    return x, y # Coordenadas em km

def xy_para_latlon(x, y, lat_ref, lon_ref, R=RAIO_TERRA_KM):
    # Projeta coordenadas cartesianas em lat/lon com cosseno
    lat_ref_rad = np.radians(lat_ref)
    lat = lat_ref + np.degrees(y / R)
    lon = lon_ref + np.degrees(x / (R * np.cos(lat_ref_rad)))
    return lat, lon # Coordenadas em graus

#-Funções-de-Plotagem-dos-Mapas──────────────────────────────────────

_cache_poligonos: dict = {}

def _carregar_poligonos_mapa(caminho: str) -> list:

    with open(caminho, encoding="utf-8") as f:
        geojson = json.load(f)
    poligonos = []
    for feat in geojson["features"]:
        geom = feat["geometry"]
        partes = (
            geom["coordinates"]
            if geom["type"] == "MultiPolygon"
            else [geom["coordinates"]]
        )
        poligonos.extend(partes)
    return poligonos

def adicionar_mapa_base(
    ax,
    caminho: str = "dataset/mapa_base_atlantico.geojson",
    cor_terra: str = "#e2e2e2",
    cor_contorno: str = "#aaaaaa",
    margem_graus: float = 2.0,
) -> None:

    if caminho not in _cache_poligonos:
        _cache_poligonos[caminho] = _carregar_poligonos_mapa(caminho)
    poligonos = _cache_poligonos[caminho]

    ax.set_aspect("equal")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    lon_min, lon_max = xlim[0] - margem_graus, xlim[1] + margem_graus
    lat_min, lat_max = ylim[0] - margem_graus, ylim[1] + margem_graus

    for poligono in poligonos:
        anel_externo = poligono[0]
        lons = [p[0] for p in anel_externo]
        lats = [p[1] for p in anel_externo]
        if (
            max(lons) < lon_min
            or min(lons) > lon_max
            or max(lats) < lat_min
            or min(lats) > lat_max
        ):
            continue  # polígono fora da área visível
        vertices, codigos = [], []
        for anel in poligono:
            vertices += anel
            codigos += (
                [Path.MOVETO]
                + [Path.LINETO] * (len(anel) - 2)
                + [Path.CLOSEPOLY]
            )
        ax.add_patch(
            PathPatch(
                Path(vertices, codigos),
                facecolor=cor_terra,
                edgecolor=cor_contorno,
                linewidth=0.4,
                zorder=0,
            )
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#-Funções-de-Calculos-Geométricos────────────────────────────────────

def comprimento(p1, p2) -> float:
    # Comprimento euclidiano (km) do segmento p1-p2
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

def componentes_distancia(li_inicio, li_fim, lj_inicio, lj_fim):
    """
    Função de distância entre segmentos de reta do algoritmo TRACLUS
    Li = segmento mais longo (referência)
    Lj = segmento a medir
    """

    # Calcula vetor direcional e comprimento
    li_inicio, li_fim, lj_inicio, lj_fim = map(
        np.asarray, (li_inicio, li_fim, lj_inicio, lj_fim)
    )
    v_li = li_fim - li_inicio
    comp_li = np.hypot(*v_li)
    v_lj = lj_fim - lj_inicio
    comp_lj = np.hypot(*v_lj)

    if comp_li < EPSILON:
        # Li degenerado (ponto): projeta Lj sobre o ponto li_inicio
        l_perp1 = np.hypot(*(lj_inicio - li_inicio))
        l_perp2 = np.hypot(*(lj_fim - li_inicio))
        soma = l_perp1 + l_perp2
        d_perp = (l_perp1**2 + l_perp2**2) / soma if soma > EPSILON else 0.0
        return float(d_perp), 0.0, float(comp_lj)

    # Projeções de s_j e e_j sobre Li
    u1 = ((lj_inicio - li_inicio) @ v_li) / comp_li**2
    u2 = ((lj_fim - li_inicio) @ v_li) / comp_li**2
    ps = li_inicio + u1 * v_li
    pe = li_inicio + u2 * v_li

    # Distância perpendicular
    l_perp1 = np.hypot(*(lj_inicio - ps))
    l_perp2 = np.hypot(*(lj_fim - pe))
    soma_perp = l_perp1 + l_perp2
    d_perp = (
        (l_perp1**2 + l_perp2**2) / soma_perp if soma_perp > EPSILON else 0.0
    )

    # Distância paralela
    l_par1 = min(
        np.hypot(*(ps - li_inicio)), np.hypot(*(ps - li_fim))
    )
    l_par2 = min(
        np.hypot(*(pe - li_inicio)), np.hypot(*(pe - li_fim))
    )
    d_par = min(l_par1, l_par2)

    # Distância angular
    if comp_lj < EPSILON:
        cos_theta = 0.0
    else:
        cos_theta = np.clip(
            (v_li @ v_lj) / (comp_li * comp_lj), -1.0, 1.0
        )
    theta_graus = np.degrees(np.arccos(cos_theta))
    d_theta = (
        comp_lj * np.sin(np.radians(theta_graus))
        if theta_graus < 90
        else comp_lj
    )
    # Retorna a dist. perpendicular, paralela e theta entre Li e Lj
    return float(d_perp), float(d_par), float(d_theta)