# Universidade Estadual de Londrina

**Curso:** Ciência de Dados e Inteligência Artificial

**Disciplina:** 2COP020 - Aprendizado de Máquina Não Supervisionado

**Docente:** Gustavo Taiji Naozuka

**Alunos:** Adriano Lucio Uchoa Brandao\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Herik Daurizio Ricardo\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Julia Yokoyama Massaki\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sofia Gutschow Casal

# Clusterização de Trajetórias de Ciclones Tropicais com TRACLUS

> Aplicação do algoritmo TRACLUS (*Trajectory Clustering: A Partition-and-Group Framework*) a trajetórias de tempestades e furacões do Oceano Atlântico, utilizando dados do HURDAT2 (2004–2025).

---

## Introdução

Este projeto implementa o **TRACLUS**, um algoritmo de clusterização de trajetórias baseado no framework de **partição e agrupamento**. Em vez de comparar trajetórias inteiras, o TRACLUS decompõe cada trajetória em segmentos de reta por meio do princípio **MDL** (*Minimum Description Length*) e agrupa segmentos similares entre trajetórias distintas. O resultado são **sub-trajetórias comuns** padrões de deslocamento que passariam despercebidos ao se comparar trajetórias na íntegra.

O algoritmo é aplicado a 215 trajetórias de tempestades tropicais e furacões do Atlântico entre 2004 e 2025, extraídas da base **HURDAT2** (NOAA/NHC). O objetivo é identificar regimes de deslocamento recorrentes a partir exclusivamente da geometria das trajetórias, sem utilizar nenhuma variável meteorológica como entrada, e depois cruzar os clusters obtidos com intensidade, duração e sazonalidade para interpretá-los fisicamente.

O pipeline completo é dividido em duas fases principais:

- **Fase 1 - Partição (MDL):** cada trajetória é aproximada por um conjunto mínimo de segmentos de reta, equilibrando *conciseness* (poucos segmentos) e *preciseness* (boa aproximação).
- **Fase 2 - Agrupamento (DBSCAN adaptado):** os segmentos de todas as trajetórias são agrupados por densidade usando uma distância composta pelas componentes perpendicular ($d_\perp$), paralela ($d_\parallel$) e angular ($d_\theta$). Cada cluster com cardinalidade mínima recebe uma trajetória representativa gerada por varredura (*sweep line*).

A avaliação dos clusters combina métricas adaptadas: coeficiente Silhouette e índice Davies-Bouldin.

---

## Estrutura do Notebook

| Bloco | Conteúdo |
|---|---|
| **1. Introdução** | Contexto do problema, visão geral do TRACLUS e suas duas fases |
| **2. Dados e pré-processamento** | Carregamento do HURDAT2, critérios de filtragem, EDA e conversão de coordenadas geográficas para cartesianas |
| **3. Fase 1 - Partição (MDL)** | Distâncias entre segmentos, custo MDL, algoritmo de partição e validação visual |
| **4. Fase 2 - Agrupamento** | Matriz de distâncias, heurística de $\varepsilon$ por entropia, busca em grade e clusterização final com trajetória representativa |
| **5. Avaliação dos clusters** | Silhouette adaptado, coesão intra-cluster, separação inter-cluster e Davies-Bouldin |
| **6. Interpretação** | Cruzamento dos clusters com intensidade, duração, velocidade translacional e sazonalidade |

---

## Tutorial de Execução

### Pré-requisitos

- Python **3.10**+
- Git

### 1. Clone o repositório

```bash
git clone https://github.com/HerikDR/Algoritmo-TRACLUS
cd traclusnotebook
```

### 2. Crie e ative o ambiente virtual

```bash
# Criação
python -m venv .venv

# Ativação — Linux/macOS
source .venv/bin/activate

# Ativação — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Inicie o Jupyter

```bash
jupyter notebook traclus_hurdat.ipynb
```

> O notebook espera encontrar os arquivos de dados no caminho `dataset/`. Certifique-se de que a estrutura de pastas está correta antes de executar (veja a seção abaixo).

### Estrutura de Pastas Esperada

```
TRACLUSNOTEBOOK/
├── .venv/
├── dataset/
│   ├── hurdat2_trajectories.csv
│   └── mapa_base_atlantico.geojson
├── .gitignore
├── README.md
├── requirements.txt
├── traclus_core.py
└── traclus_hurdat.ipynb
```

> **Atenção:** o arquivo `mapa_base_atlantico.geojson` é necessário para renderizar o mapa de fundo nos gráficos. Ele deve estar dentro da pasta `dataset/`, no mesmo nível do CSV.

---

## Fonte dos Dados

O dataset utilizado é derivado da base **HURDAT2** (*Hurricane Database 2*), mantida pelo **National Hurricane Center (NOAA)**. Ela registra a posição e a intensidade de todos os sistemas tropicais do Atlântico a cada 6 horas, desde 1851.

- **Acesse o dataset original em:** [Clique aqui](https://www.nhc.noaa.gov/data/#hurdat)

### Características do Dataset Pré-processado (`hurdat2_trajectories.csv`)

| Característica | Valor |
|---|---|
| Período coberto | 2004–2025 |
| Total de registros (observações de 6h) | 11.885 |
| Sistemas (storm_id) únicos | 388 |
| Tipos de status disponíveis | TD, TS, HU, EX, LO, SS, SD, DB, WV |
| Vento máximo registrado | 165 kt |
| Pressão mínima registrada | 882 hPa |
| **Após filtragem (TS/HU, ≥ 12 obs.)** | |
| Trajetórias válidas | 215 |
| Registros filtrados | 5.882 |

### Colunas do Dataset

| Coluna | Descrição |
|---|---|
| `storm_id` | Identificador único do sistema (ex.: `AL012004`) |
| `storm_name` | Nome do sistema |
| `datetime`, `year`, `month`, `day`, `hour`, `minute` | Data e hora da observação |
| `record_id` | Marcador de eventos especiais (landfall etc.) |
| `status` | Classificação do sistema na observação |
| `latitude`, `longitude` | Posição do centro do sistema (graus) |
| `wind_kt` | Vento máximo sustentado (nós) |
| `pressure_mb` | Pressão central mínima (hPa) |

---

## Referências

AGGARWAL, Charu C.; REDDY, Chandan K. (ed.). **Data clustering**: algorithms and applications. Boca Raton: CRC Press, 2014. ISBN 978-1-4665-5821-2.

DAVIES, David L.; BOULDIN, Donald W. A cluster separation measure. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, [*S. l.*], v. PAMI-1, n. 2, p. 224-227, 1979. DOI: 10.1109/TPAMI.1979.4766909.

ESTER, Martin; KRIEGEL, Hans-Peter; SANDER, Jörg; XU, Xiaowei. A density-based algorithm for discovering clusters in large spatial databases with noise. **In**: INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 2., 1996, Portland. **Proceedings [...]**. Menlo Park: AAAI Press, 1996. p. 226-231.

LANDSEA, Christopher W.; FRANKLIN, James L. Atlantic hurricane database uncertainty and presentation of a new database format. **Monthly Weather Review**, Boston, v. 141, n. 10, p. 3576-3592, 2013. DOI: 10.1175/MWR-D-12-00254.1.

LEE, Jae-Gil; HAN, Jiawei; WHANG, Kyu-Young. Trajectory clustering: a partition-and-group framework. **In**: ACM SIGMOD INTERNATIONAL CONFERENCE ON MANAGEMENT OF DATA, 2007, Beijing. **Proceedings of the 2007 ACM SIGMOD International Conference on Management of Data (SIGMOD '07)**. New York: ACM, 2007. p. 593-604. DOI: 10.1145/1247480.1247546.

NATIONAL HURRICANE CENTER. **NHC data archive**: HURDAT2 - Atlantic hurricane database. Miami: NOAA, c2026. Disponível em: https://www.nhc.noaa.gov/data/. Acesso em: 21 jun. 2026.

ROUSSEEUW, Peter J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. **Journal of Computational and Applied Mathematics**, Amsterdam, v. 20, n. 1, p. 53-65, 1987. DOI: 10.1016/0377-0427(87)90125-7.
