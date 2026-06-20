"""
=====================================================================
traclus_core.py
Implementação do núcleo do pipeline TRACLUS
=====================================================================
"""
# Bibliotecas
import json
from collections import deque

import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

#-Constantes-Globais-================================================

RAIO_TERRA_KM = 6371.0 # Raio da terra para projeção equirretangular
EPSILON = 1e-6         # Piso numério para evitar log2(o) e divisão por 0
NAO_CLASSIFICADO = -2  # Rótulo para segmentos não visitados
RUIDO = -1             # Rótulo para segmentos de ruído

#-Funções-de-Projeção-Geográfica-====================================

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

#-Funções-de-Plotagem-dos-Mapas-=====================================

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

#-Funções-de-Calculos-Geométricos-===================================

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
        # Retorna a distância degenerada
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

def distancia_segmentos(segmento_a, segmento_b, pesos=(1.0, 1.0, 1.0)) -> float:
    """
    Compara o comprimentos dos dois segmentos e identifica Li e Lj
    chama componentes_distancia e calcula as distâncias
    """
    a_inicio, a_fim = segmento_a
    b_inicio, b_fim = segmento_b
    comp_a = comprimento(a_inicio, a_fim)
    comp_b = comprimento(b_inicio, b_fim)
    if comp_a >= comp_b:
        li_inicio, li_fim, lj_inicio, lj_fim = a_inicio, a_fim, b_inicio, b_fim
    else:
        li_inicio, li_fim, lj_inicio, lj_fim = b_inicio, b_fim, a_inicio, a_fim
    d_perp, d_par, d_theta = componentes_distancia(
        li_inicio, li_fim, lj_inicio, lj_fim
    )
    w_perp, w_par, w_theta = pesos
    # Retorna o resumo da distância entre os dois segmentos
    return w_perp * d_perp + w_par * d_par + w_theta * d_theta

#-Funções-de-Custo-(MDL)-============================================

def custo_L(x: float) -> float:
    # L(x) = log2(x)
    return np.log2(max(x, EPSILON))

def mdl_com_particao(pontos, i: int, j: int) -> float:
    """
    Calcula o custo MDL de representar o trecho da trajetória partida
    Corresponde a L(H) + L(D|H)
    """
    p_i, p_j = pontos[i], pontos[j]
    custo = custo_L(comprimento(p_i, p_j))   # L(H)
    for k in range(i, j):                    # L(D|H)
        d_perp, _, d_theta = componentes_distancia(
            p_i, p_j, pontos[k], pontos[k + 1]
        )
        custo += custo_L(d_perp) + custo_L(d_theta)
    # Retorna o custo MDL com partição
    return custo

def mdl_sem_particao(pontos, i: int, j: int, vies_supressao: float = 0.0) -> float:
    """
    Calcula o custo MDL de representar o trecho da trajetória preservada
    O viés de supressão é somado ao custo para suprimir partições curtas
    """
    custo = 0.0
    for k in range(i, j):
        custo += custo_L(comprimento(pontos[k], pontos[k + 1]))
    # Retorna o custo MDL sem partição
    return custo + vies_supressao

#-Função-de-Partição-(ATP)-==========================================

def particionar_trajetoria(pontos, vies_supressao: float = 0.0) -> list:
    # Algoritmo Approximate Trajectory Partitioning O(n)
    n = len(pontos)
    pontos_caracteristicos = [0]
    indice_inicio = 0
    comprimento_janela = 1

    # Percorre a trajetória expandindo a janela
    while indice_inicio + comprimento_janela <= n - 1:
        indice_atual = indice_inicio + comprimento_janela
        custo_par = mdl_com_particao(pontos, indice_inicio, indice_atual) # Calcula o custo partindo
        custo_nopar = mdl_sem_particao(                                   # Calcula o custo sem partir
            pontos, indice_inicio, indice_atual, vies_supressao
        )
        # Verifica se o custo de partir é maior que não partir
        if custo_par > custo_nopar:
            # Se sim, particione
            pontos_caracteristicos.append(indice_atual - 1)
            indice_inicio = indice_atual - 1
            comprimento_janela = 1
        else:
            # Se não, amplie a janela
            comprimento_janela += 1

    pontos_caracteristicos.append(n - 1)
    # Retorna o índice dos pontos característicos
    return pontos_caracteristicos

#-Função-de-Matriz-de-Distância-=====================================

def calcular_matriz_distancia(
    # Matriz de distância N×N entre todos os segmentos
    segmentos_df, pesos=(1.0, 1.0, 1.0), tam_bloco: int = 300
):
    # Extrai os pontos finais e iniciais de todos segmentos
    si = segmentos_df[["x_inicio", "y_inicio"]].values
    ei = segmentos_df[["x_fim", "y_fim"]].values
    v = ei - si                          # Calcula o vetor de direção
    comp = np.hypot(v[:, 0], v[:, 1])    # Calcula o comprimento
    n = len(segmentos_df)
    w_perp, w_par, w_theta = pesos

    # Inicializa a matriz em blocos
    M = np.zeros((n, n))
    for inicio in range(0, n, tam_bloco):
        fim = min(inicio + tam_bloco, n)
        si_b = si[inicio:fim]
        ei_b = ei[inicio:fim]
        v_b = v[inicio:fim]
        comp_b = comp[inicio:fim]
        comp_b2 = np.where(comp_b < EPSILON, 1.0, comp_b**2)

        # Projeções dos pontos iniciais e finais de cada segmento sobre cada Li
        diff_s = si[None, :, :] - si_b[:, None, :]
        diff_e = ei[None, :, :] - si_b[:, None, :]
        u1 = (diff_s * v_b[:, None, :]).sum(-1) / comp_b2[:, None]
        u2 = (diff_e * v_b[:, None, :]).sum(-1) / comp_b2[:, None]
        ps = si_b[:, None, :] + u1[:, :, None] * v_b[:, None, :]
        pe = si_b[:, None, :] + u2[:, :, None] * v_b[:, None, :]

        # Distância perpendicular
        l_perp1 = np.hypot(
            si[None, :, 0] - ps[:, :, 0], si[None, :, 1] - ps[:, :, 1]
        )
        l_perp2 = np.hypot(
            ei[None, :, 0] - pe[:, :, 0], ei[None, :, 1] - pe[:, :, 1]
        )
        soma_perp = l_perp1 + l_perp2
        d_perp = np.where(
            soma_perp > EPSILON,
            (l_perp1**2 + l_perp2**2) / np.where(soma_perp > EPSILON, soma_perp, 1),
            0.0,
        )

        # Distância paralela
        l_par1 = np.minimum(
            np.hypot(
                ps[:, :, 0] - si_b[:, None, 0], ps[:, :, 1] - si_b[:, None, 1]
            ),
            np.hypot(
                ps[:, :, 0] - ei_b[:, None, 0], ps[:, :, 1] - ei_b[:, None, 1]
            ),
        )
        l_par2 = np.minimum(
            np.hypot(
                pe[:, :, 0] - si_b[:, None, 0], pe[:, :, 1] - si_b[:, None, 1]
            ),
            np.hypot(
                pe[:, :, 0] - ei_b[:, None, 0], pe[:, :, 1] - ei_b[:, None, 1]
            ),
        )
        d_par = np.minimum(l_par1, l_par2)

        # Distância angular
        comp_lj = comp[None, :]
        produto = (v_b[:, None, :] * v[None, :, :]).sum(-1)
        denom = comp_b[:, None] * comp_lj
        cos_theta = np.clip(
            np.where(denom > EPSILON, produto / np.where(denom > EPSILON, denom, 1), 0.0),
            -1.0,
            1.0,
        )
        theta = np.degrees(np.arccos(cos_theta))
        d_theta = np.where(theta < 90, comp_lj * np.sin(np.radians(theta)), comp_lj)
        d_theta = np.broadcast_to(d_theta, (fim - inicio, n)).copy()

        # Li degenerado
        degenerado = comp_b < EPSILON
        if degenerado.any():
            l1d = np.hypot(
                si[None, :, 0] - si_b[:, None, 0],
                si[None, :, 1] - si_b[:, None, 1],
            )
            l2d = np.hypot(
                ei[None, :, 0] - si_b[:, None, 0],
                ei[None, :, 1] - si_b[:, None, 1],
            )
            soma_d = l1d + l2d
            d_perp_d = np.where(
                soma_d > EPSILON,
                (l1d**2 + l2d**2) / np.where(soma_d > EPSILON, soma_d, 1),
                0.0,
            )
            d_perp[degenerado] = d_perp_d[degenerado]
            d_par[degenerado] = 0.0
            d_theta[degenerado] = np.broadcast_to(comp_lj, (fim - inicio, n))[
                degenerado
            ]

        M[inicio:fim, :] = w_perp * d_perp + w_par * d_par + w_theta * d_theta

    # Simetria: segmento mais longo = Li; empate -> menor índice
    idx = np.arange(n)
    i_eh_li = (comp[:, None] > comp[None, :]) | (
        (comp[:, None] == comp[None, :]) & (idx[:, None] < idx[None, :])
    )
    matriz = np.where(i_eh_li, M, M.T)
    np.fill_diagonal(matriz, 0.0)
    # Retorna a matriz com a distância de todos os segmentos
    return matriz

#-Heurística-de-Entropia-============================================

def entropia_vizinhanca(matriz_dist, eps: float):
    # Calcula a entropia da distribuição de |N_epsilon(L)| sobre os segmentos
    contagens = (matriz_dist <= eps).sum(axis=1).astype(float)
    p = contagens / contagens.sum()                  # Normaliza as contagens
    return -np.sum(p * np.log2(p)), contagens.mean() # Shannon e média

#-Funções-do-DBSCAN-Adaptado-========================================

def clusterizar_segmentos(matriz_dist, eps: float, min_lns: int):
    """
    Algoritmo Line Segment Clustering: DBSCAN Adaptado
    para segmentos e uso de matriz de distâncias
    """
    n = matriz_dist.shape[0]
    vizinhanca = matriz_dist <= eps
    rotulos = np.full(n, NAO_CLASSIFICADO, dtype=int)
    id_cluster = 0

    for i in range(n):
        if rotulos[i] != NAO_CLASSIFICADO:
            continue
        vizinhos_i = np.where(vizinhanca[i])[0]
        if len(vizinhos_i) >= min_lns:          # L é núcleo
            rotulos[vizinhos_i] = id_cluster    # Adiciona ao Cluster
            fila = deque(v for v in vizinhos_i if v != i)
            while fila:
                m = fila.popleft()
                vizinhos_m = np.where(vizinhanca[m])[0]
                if len(vizinhos_m) >= min_lns:
                    for x in vizinhos_m:
                        if rotulos[x] in (NAO_CLASSIFICADO, RUIDO):
                            era_novo = rotulos[x] == NAO_CLASSIFICADO
                            rotulos[x] = id_cluster
                            if era_novo:
                                fila.append(x)
            id_cluster += 1
        else:
            rotulos[i] = RUIDO
    # Retorna o vetor de rótulos dos segmentos
    return rotulos

def filtrar_cardinalidade_trajetorias(rotulos, ids_tempestade, limiar: int):
    # Descarta os clusters com trajetórias menor que o limiar
    rotulos = rotulos.copy()
    for c in np.unique(rotulos[rotulos >= 0]):
        membros = rotulos == c
        if len(set(ids_tempestade[membros])) < limiar:
            rotulos[membros] = RUIDO # Atribui como ruído
    # Retorna uma cópia dos rótulos com os clusters inválidos
    return rotulos

#-Métrica-QMeasure-==================================================

def qmeasure(matriz_dist, rotulos) -> float:
    # SSE total ponderado + penalidade de ruído
    total = 0.0

    # Soma o erro quadrático para cada cluster válido
    for c in np.unique(rotulos[rotulos >= 0]):
        idx = np.where(rotulos == c)[0]
        sub = matriz_dist[np.ix_(idx, idx)]
        total += (sub**2).sum() / (2 * len(idx))
    idx_ruido = np.where(rotulos == RUIDO)[0]

    # Penaliza o ruído
    if len(idx_ruido) > 0:
        sub = matriz_dist[np.ix_(idx_ruido, idx_ruido)]
        total += (sub**2).sum() / (2 * len(idx_ruido))
    # Retorna o qmeasure total
    return total

#-Trajetória-Representativa-=========================================

def vetor_direcao_medio(segmentos_cluster):
    # Média do vetor de direção da trajetória (não normalizado)
    vetores = (
        segmentos_cluster[["x_fim", "y_fim"]].values
        - segmentos_cluster[["x_inicio", "y_inicio"]].values
    )
    return vetores.mean(axis=0)

def gerar_trajetoria_representativa(segmentos_cluster, min_lns: int, gamma: float):
    # Algoritmo Representative Trajectory Generation

    # Calcula o vetor de direção média
    v_medio = vetor_direcao_medio(segmentos_cluster)
    norma_v = np.hypot(*v_medio)
    # Se for quase nulo, não representa
    if norma_v < EPSILON:
        return np.empty((0, 2))

    # Alinha os pontos iniciais e finais
    cos_phi, sin_phi = v_medio[0] / norma_v, v_medio[1] / norma_v
    R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    pts_inicio = segmentos_cluster[["x_inicio", "y_inicio"]].values @ R.T
    pts_fim = segmentos_cluster[["x_fim", "y_fim"]].values @ R.T

    segmentos_rotacionados, todos_x = [], []
    for (x1, y1), (x2, y2) in zip(pts_inicio, pts_fim):
        segmentos_rotacionados.append((min(x1, x2), max(x1, x2), x1, y1, x2, y2))
        todos_x += [x1, x2]
    pontos_x = sorted(set(todos_x))

    max_ativos = max(
        sum(
            1
            for xmin, xmax, *_ in segmentos_rotacionados
            if xmin - 1e-9 <= x <= xmax + 1e-9
        )
        for x in pontos_x
    )
    limiar = min(min_lns, max_ativos)

    trajetoria_rotacionada, ultimo_x = [], None
    # Varre os pontos extremos ordenados
    for x in pontos_x:
        ativos = []
        for xmin, xmax, x1, y1, x2, y2 in segmentos_rotacionados:
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                if abs(x2 - x1) < EPSILON:
                    y_interp = (y1 + y2) / 2
                else:
                    t = (x - x1) / (x2 - x1)
                    y_interp = y1 + t * (y2 - y1)
                ativos.append(y_interp)
        if len(ativos) >= limiar and (
            ultimo_x is None or (x - ultimo_x) >= gamma
        ):
            trajetoria_rotacionada.append((x, np.mean(ativos)))
            ultimo_x = x

    trajetoria_rotacionada = np.array(trajetoria_rotacionada)
    # Retorna vazio para trajetórias pequenas
    if len(trajetoria_rotacionada) == 0:
        return trajetoria_rotacionada
    # Desfaz a rotação e retorna ao sistema original
    return trajetoria_rotacionada @ R

#-Perfil-Metereológico-==============================================

def construir_perfil_meteorologico(segmentos, resumo_tempestades):
    # Constrói um Dataframe agregado por tempestade e cluster + características
    registros = []
    segs_cl = segmentos[segmentos["cluster"] >= 0].copy()
    segs_cl["dt_inicio"] = pd.to_datetime(segs_cl["datetime_inicio"])
    segs_cl["dt_fim"] = pd.to_datetime(segs_cl["datetime_fim"])
    segs_cl["dur_h_seg"] = (
        segs_cl["dt_fim"] - segs_cl["dt_inicio"]
    ).dt.total_seconds() / 3600
    segs_cl["vel_kmh"] = np.where(
        segs_cl["dur_h_seg"] > 0,
        segs_cl["comprimento_km"] / segs_cl["dur_h_seg"],
        np.nan,
    )
    segs_cl["mes"] = segs_cl["dt_inicio"].dt.month

    for (id_traj, cl), grupo in segs_cl.groupby(["id_tempestade", "cluster"]):
        rt = resumo_tempestades[resumo_tempestades["storm_id"] == id_traj]
        if len(rt) == 0:
            continue
        rt = rt.iloc[0]
        mes_p = grupo["mes"].mode()
        registros.append(
            dict(
                id_tempestade=id_traj,
                cluster=cl,
                n_segmentos_no_cluster=len(grupo),
                vento_medio_no_cluster=grupo["vento_medio"].mean(),
                vento_max_no_cluster=grupo["vento_max"].max(),
                vento_max_tempestade=rt["vento_max"],
                duracao_dias=rt["duracao_dias"],
                ano=rt["ano"],
                velocidade_media_kmh=grupo["vel_kmh"].mean(),
                mes_pico=int(mes_p.iloc[0]) if len(mes_p) > 0 else np.nan,
            )
        )
    return pd.DataFrame(registros)

#====================================================================