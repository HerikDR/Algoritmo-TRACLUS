# Clusterização de Trajetórias de Ciclones Tropicais com TRACLUS

## Estudo de caso com dados HURDAT2 - Oceano Atlântico (2004–2025)

---

Este notebook implementa o **TRACLUS** (TRAjectory CLUStering) e o aplica a trajetórias de tempestades e furacões do Atlântico da base **HURDAT2** (NOAA/NHC). O algoritmo usa um framework de **partição-e-agrupamento**: cada trajetória é quebrada em segmentos de reta, e segmentos semelhantes de tempestades diferentes ou de trechos diferentes da mesma tempestade são agrupados por densidade, revelando **sub-trajetórias comuns** que passariam despercebidas ao comparar trajetórias inteiras.

**Dados:** `hurdat2_trajectories.csv` (posição, `wind_kt`, `pressure_mb` e status a cada 6h, Atlântico 2004–2025).

## 1.1 Motivação: por que particionar trajetórias?

Tempestades que recurvam para nordeste em latitudes médias e tempestades que seguem retas para oeste em baixas latitudes podem ter origens completamente diferentes. Comparando **trajetórias inteiras**, duas tempestades que só compartilham o trecho final de recurvatura dificilmente seriam agrupadas — a distância total seria dominada pelas partes diferentes.

Por isso particionamos cada trajetória em **segmentos de reta** e agrupamos por proximidade geométrica: isso isola **comportamentos locais comuns** (ex.: recurvatura ao atingir a costa leste dos EUA), mesmo quando as trajetórias completas são 
muito distintas — especialmente útil para previsão de landfall.

## 1.2 Definições formais

- **Trajetória**: sequência ordenada $TR = p_1 p_2 \dots p_{len}$, com cada $p_j$ um ponto (lat/lon, depois convertido para km).
- **Partição de trajetória**: segmento de reta $p_i p_j$ ($i<j$) entre dois **pontos característicos** consecutivos; cada trajetória vira uma sequência de partições.
- **Cluster**: conjunto de partições (possivelmente de trajetórias diferentes) mutuamente próximas segundo uma função de distância entre segmentos.
- **Trajetória representativa**: trajetória "imaginária" que resume o comportamento comum de um cluster — a materialização da sub-trajetória comum.

O TRACLUS recebe $\mathcal{I} = \{TR_1, \dots, TR_{n}\}$ e devolve clusters $\mathcal{O} = \{C_1, \dots, C_{k}\}$, cada um com sua trajetória representativa.

## 1.3 Visão geral do algoritmo TRACLUS

O TRACLUS tem **duas fases**:

**Fase 1 — Partição.** Cada trajetória é decomposta em segmentos de reta pelo princípio **MDL**, equilibrando *conciseness* (poucos segmentos) e *preciseness* (boa aproximação). Um algoritmo aproximado $O(n)$ escolhe pontos característicos sempre que particionar fica mais barato, em bits, do que manter a trajetória original.

**Fase 2 — Agrupamento.** Todos os segmentos formam um conjunto único $\mathcal{D}$, clusterizado por densidade (estilo DBSCAN) com uma distância composta pelas componentes perpendicular ($d_\perp$), paralela ($d_\parallel$) e angular ($d_\theta$). Clusters válidos (cardinalidade mínima de trajetórias) recebem uma trajetória representativa via varredura (*sweep line*).