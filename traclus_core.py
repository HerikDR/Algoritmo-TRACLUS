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

