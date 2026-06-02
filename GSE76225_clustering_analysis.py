"""
================================================================================
GSE76225 / GSE76226  –  Análise de Clustering Transcriptômico
================================================================================
Estudo: Identificação de endótipos de asma T2HIGH, T17HIGH e T2LOW/T17LOW
         usando 8 genes assinatura em epitélio nasal.

Datasets:
  - GSE76226  (Epithelial Brushing) : 1__Matrix_for_clustering.csv   [n=91]
  - GSE76225  (Bronchial Biopsy)    : 1_GSE76225clustering_matrix.csv [n=99]

Estrutura de pareamento:
  - 91 pacientes têm AMBOS os tecidos coletados (brushing + biopsy)
  - 8 pacientes têm apenas biópsia (GSM1977244–GSM1977251)
  - Lógica de par: GSM_brushing + 91 = GSM_biopsy (offset constante)

Referência:
  Yue M, Gaietto K, Han YY, et al. Transcriptomic profiles in nasal
  epithelium and asthma endotypes in youth. JAMA. 2025.
  doi:10.1001/jama.2024.22684

Metodologia expandida:
  5 métodos de clustering (K-means, Hierárquico Ward, K-medoids PAM,
  Mistura Gaussiana, Espectral) com validação interna (Silhueta,
  Calinski-Harabasz, Davies-Bouldin), ARI pairwise, consenso e
  estatística Gap para seleção de k.
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    pairwise_distances,
)
from scipy.stats import spearmanr, pearsonr, ks_2samp, wilcoxon, ttest_rel
from scipy.cluster.hierarchy import dendrogram, linkage

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURAÇÃO GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
GENES     = ["CLCA1", "CSF3", "CXCL1", "CXCL2", "CXCL3", "IL8", "POSTN", "SERPINB2"]
T2_GENES  = ["POSTN", "SERPINB2", "CLCA1"]          # assinatura T2 (3 genes)
T17_GENES = ["CSF3", "CXCL1", "CXCL2", "CXCL3", "IL8"]  # assinatura T17 (5 genes)
GENE_ORDER_HEATMAP = ["POSTN", "SERPINB2", "CLCA1",
                      "CSF3", "CXCL1", "CXCL2", "CXCL3", "IL8"]

K         = 3           # número de clusters (endótipos)
SEED      = 42          # semente para reprodutibilidade
N_INIT_KM = 50          # inicializações K-means (estabilidade)
N_REFS_GAP= 20          # referências bootstrap para Gap statistic
K_MAX     = 8           # máximo de k testado na seleção

METHODS   = ["K-means", "Hierarchical", "K-medoids", "Gaussian Mixture", "Spectral"]
PROFILES  = ["T2HIGH / T17LOW", "T2LOW / T17HIGH", "T2LOW / T17LOW"]

PALETTE   = {
    "T2HIGH / T17LOW" : "#c0392b",
    "T2LOW / T17HIGH" : "#2471a3",
    "T2LOW / T17LOW"  : "#1e8449",
}

# Caminhos
INPUT_BIOPSY   = "/mnt/user-data/uploads/1_GSE76225clustering_matrix.csv"
INPUT_BRUSHING = "/mnt/user-data/uploads/1__Matrix_for_clustering.csv"
OUTDIR         = "/mnt/user-data/outputs/"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family"    : "DejaVu Sans",
    "font.size"      : 10,
    "axes.titlesize" : 12,
    "axes.labelsize" : 11,
    "figure.dpi"     : 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("CARREGANDO DADOS")
print("=" * 70)

biopsy_raw   = pd.read_csv(INPUT_BIOPSY).set_index("Sample_ID")
brushing_raw = pd.read_csv(INPUT_BRUSHING).set_index("ID")

X_biopsy   = biopsy_raw[GENES].copy()
X_brushing = brushing_raw[GENES].copy()

print(f"  GSE76225 – Bronchial Biopsy   : {X_biopsy.shape[0]} amostras × {X_biopsy.shape[1]} genes")
print(f"  GSE76226 – Epithelial Brushing: {X_brushing.shape[0]} amostras × {X_brushing.shape[1]} genes")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  IDENTIFICAÇÃO DAS AMOSTRAS PAREADAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANÁLISE DE PAREAMENTO")
print("=" * 70)

nums_biopsy   = [int(x.replace("GSM", "")) for x in X_biopsy.index]
nums_brushing = [int(x.replace("GSM", "")) for x in X_brushing.index]

# Verificação do offset constante
offsets = [nums_biopsy[i] - nums_brushing[i]
           for i in range(min(len(nums_biopsy), len(nums_brushing)))]
unique_offsets = set(offsets)

if len(unique_offsets) == 1:
    OFFSET = list(unique_offsets)[0]
    print(f"\n  Offset GSM constante detectado: +{OFFSET}")
    print(f"  → GSM_biopsy = GSM_brushing + {OFFSET}")
    print(f"  → Pareamento posicional confirmado para as primeiras {len(nums_brushing)} amostras")
else:
    print("  AVISO: Offset não é constante – verificar manualmente.")
    OFFSET = None

N_PAIRED   = len(X_brushing)   # 91  (todos do brushing têm par no biopsy)
N_UNPAIRED = len(X_biopsy) - N_PAIRED  # 8  (biopsy sem brushing correspondente)

paired_biopsy_ids   = list(X_biopsy.index[:N_PAIRED])
unpaired_biopsy_ids = list(X_biopsy.index[N_PAIRED:])
paired_brushing_ids = list(X_brushing.index)

print(f"\n  Amostras pareadas (ambos os tecidos): {N_PAIRED}")
print(f"  Amostras biopsy-only (sem brushing):  {N_UNPAIRED}")
print(f"\n  IDs biopsy sem par brushing correspondente:")
for sid in unpaired_biopsy_ids:
    print(f"    {sid}")

# Tabela de pares
pair_table = pd.DataFrame({
    "Patient_Index"        : range(1, N_PAIRED + 1),
    "ID_Brushing_GSE76226" : paired_brushing_ids,
    "ID_Biopsy_GSE76225"   : paired_biopsy_ids,
    "Paired"               : True,
})
# Adicionar os biopsy-only
biopsy_only_df = pd.DataFrame({
    "Patient_Index"        : range(N_PAIRED + 1, N_PAIRED + N_UNPAIRED + 1),
    "ID_Brushing_GSE76226" : None,
    "ID_Biopsy_GSE76225"   : unpaired_biopsy_ids,
    "Paired"               : False,
})
pair_table = pd.concat([pair_table, biopsy_only_df], ignore_index=True)

print(f"\n  Tabela de pareamento (primeiros 10 pares):")
print(pair_table.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PRÉ-PROCESSAMENTO: PADRONIZAÇÃO Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PRÉ-PROCESSAMENTO – Z-SCORE PADRONIZAÇÃO")
print("=" * 70)

scaler_biopsy   = StandardScaler()
scaler_brushing = StandardScaler()

X_biopsy_sc   = pd.DataFrame(
    scaler_biopsy.fit_transform(X_biopsy),
    columns=GENES, index=X_biopsy.index
)
X_brushing_sc = pd.DataFrame(
    scaler_brushing.fit_transform(X_brushing),
    columns=GENES, index=X_brushing.index
)

print("  Padronização Z-score independente por dataset aplicada.")
print("  (garante média=0, SD=1 por gene em cada coorte)")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def assign_profile(labels_arr, Xs):
    """
    Anota clusters como T2HIGH/T17LOW, T2LOW/T17HIGH ou T2LOW/T17LOW
    com base nos centróides das assinaturas T2 e T17.
    """
    df = Xs.copy()
    df["__label"] = labels_arr
    centroids = df.groupby("__label").mean()

    t2_score  = centroids[T2_GENES].mean(axis=1)
    cl_t2h    = t2_score.idxmax()

    remaining  = [c for c in centroids.index if c != cl_t2h]
    t17_score  = centroids[T17_GENES].mean(axis=1)
    cl_t17h    = t17_score.loc[remaining].idxmax()

    cl_low = [c for c in centroids.index
              if c not in [cl_t2h, cl_t17h]][0]

    mapping = {
        cl_t2h  : "T2HIGH / T17LOW",
        cl_t17h : "T2LOW / T17HIGH",
        cl_low  : "T2LOW / T17LOW",
    }
    return np.array([mapping[l] for l in labels_arr])


def run_kmedoids(Xs, k=3, seed=SEED, max_iter=300):
    """
    Implementação PAM (Partitioning Around Medoids) via distâncias pairwise.
    Mais robusto a outliers que K-means.
    """
    D = pairwise_distances(Xs)
    rng = np.random.RandomState(seed)
    medoid_idx = list(rng.choice(len(Xs), k, replace=False))
    for _ in range(max_iter):
        dists = D[:, medoid_idx]
        labels = dists.argmin(axis=1)
        new_medoids = []
        for c in range(k):
            mask = labels == c
            if not mask.any():
                new_medoids.append(medoid_idx[c])
                continue
            sub = D[np.ix_(mask, mask)]
            new_medoids.append(np.where(mask)[0][sub.sum(axis=1).argmin()])
        if sorted(new_medoids) == sorted(medoid_idx):
            break
        medoid_idx = new_medoids
    return labels


def run_all_methods(Xs, k=K):
    """
    Executa os 5 métodos de clustering e retorna labels brutas (inteiros).
    """
    results = {}

    # K-means
    km = KMeans(n_clusters=k, random_state=SEED, n_init=N_INIT_KM)
    results["K-means"] = km.fit_predict(Xs)

    # Hierárquico Ward
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    results["Hierarchical"] = hc.fit_predict(Xs)

    # K-medoids (PAM)
    results["K-medoids"] = run_kmedoids(Xs, k=k)

    # Mistura Gaussiana
    gm = GaussianMixture(n_components=k, random_state=SEED, n_init=20,
                         covariance_type="full")
    results["Gaussian Mixture"] = gm.fit_predict(Xs)

    # Espectral
    sp = SpectralClustering(n_clusters=k, random_state=SEED,
                            affinity="rbf", n_init=20)
    results["Spectral"] = sp.fit_predict(Xs)

    return results


def build_consensus(profile_dict, methods):
    """
    Perfil de consenso: rótulo majoritário entre todos os métodos.
    Retorna (labels_consenso, fração_de_acordo [0..1]).
    """
    arr = np.array([profile_dict[m] for m in methods]).T  # (n, n_methods)
    labels, agreement = [], []
    for row in arr:
        vals, counts = np.unique(row, return_counts=True)
        majority = vals[counts.argmax()]
        labels.append(majority)
        agreement.append(counts.max() / len(methods))
    return np.array(labels), np.array(agreement)


def compute_internal_metrics(Xs, raw_labels_dict):
    """Silhueta, Calinski-Harabász, Davies-Bouldin para cada método."""
    rows = []
    for m, lbl in raw_labels_dict.items():
        rows.append({
            "Método"            : m,
            "Silhueta ↑"        : silhouette_score(Xs, lbl),
            "Calinski-Harabász ↑": calinski_harabasz_score(Xs, lbl),
            "Davies-Bouldin ↓"  : davies_bouldin_score(Xs, lbl),
        })
    return pd.DataFrame(rows).set_index("Método")


def ari_matrix(raw_labels_dict, methods):
    """Adjusted Rand Index pairwise entre todos os métodos."""
    n = len(methods)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = adjusted_rand_score(raw_labels_dict[methods[i]],
                                    raw_labels_dict[methods[j]])
            mat[i, j] = mat[j, i] = v
    return pd.DataFrame(mat, index=methods, columns=methods)


def elbow_gap(Xs, k_max=K_MAX, n_refs=N_REFS_GAP):
    """
    Calcula inércia (elbow), silhueta média e estatística Gap vs k.
    """
    Ks = range(2, k_max + 1)
    inertias, sils, gaps, gap_stds = [], [], [], []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=30).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))
        ref_disps = []
        for _ in range(n_refs):
            ref = np.random.uniform(Xs.min().values,
                                    Xs.max().values, Xs.shape)
            km_r = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(ref)
            ref_disps.append(np.log(km_r.inertia_))
        gaps.append(np.mean(ref_disps) - np.log(km.inertia_))
        gap_stds.append(np.std(ref_disps) * np.sqrt(1 + 1 / n_refs))
    return (list(Ks), np.array(inertias), np.array(sils),
            np.array(gaps), np.array(gap_stds))

# ─────────────────────────────────────────────────────────────────────────────
# 6.  ANÁLISE INTER-DATASET: CORRELAÇÃO E TESTE KS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CORRELAÇÃO INTER-DATASET (GSE76226 vs GSE76225)")
print("=" * 70)

mean_brushing = X_brushing.mean()
mean_biopsy   = X_biopsy.mean()

r_pear, p_pear = pearsonr(mean_brushing, mean_biopsy)
r_spear, p_spear = spearmanr(mean_brushing, mean_biopsy)

print(f"\n  Correlação das médias gênicas:")
print(f"    Pearson  r = {r_pear:.4f}  (p = {p_pear:.4f})")
print(f"    Spearman ρ = {r_spear:.4f}  (p = {p_spear:.4f})")

print(f"\n  Teste KS por gene (distribuições DS1 vs DS2):")
ks_table = []
for g in GENES:
    stat, pval = ks_2samp(X_biopsy[g], X_brushing[g])
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ks_table.append({"Gene": g, "KS stat": stat, "p-valor": pval, "Sig": sig})
    print(f"    {g:12s}  KS={stat:.3f}  p={pval:.4f}  {sig}")

ks_df = pd.DataFrame(ks_table)

# Teste de Wilcoxon nas amostras PAREADAS (n=91)
print(f"\n  Teste de Wilcoxon (pareado, n={N_PAIRED}) por gene:")
wilcox_table = []
for g in GENES:
    bru = X_brushing[g].values
    bio = X_biopsy.iloc[:N_PAIRED][g].values
    stat, pval = wilcoxon(bru, bio)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    wilcox_table.append({"Gene": g, "W stat": stat, "p-valor": pval, "Sig": sig})
    print(f"    {g:12s}  W={stat:.1f}  p={pval:.4f}  {sig}")

wilcox_df = pd.DataFrame(wilcox_table)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SELEÇÃO DO k ÓTIMO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SELEÇÃO DO k ÓTIMO (Elbow + Silhueta + Gap Statistic)")
print("=" * 70)

print("  Calculando para GSE76226 (Brushing)...")
Ks_b, iner_b, sil_b, gap_b, gstd_b = elbow_gap(X_brushing_sc)
print("  Calculando para GSE76225 (Biopsy)...")
Ks_bp, iner_bp, sil_bp, gap_bp, gstd_bp = elbow_gap(X_biopsy_sc)

print(f"\n  Brushing – Silhueta máx em k={Ks_b[np.argmax(sil_b)]}")
print(f"  Biopsy   – Silhueta máx em k={Ks_bp[np.argmax(sil_bp)]}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLUSTERING: 5 MÉTODOS × 2 DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLUSTERING (k=3, 5 MÉTODOS)")
print("=" * 70)

print("  Rodando métodos no GSE76226 (Brushing)...")
raw_b  = run_all_methods(X_brushing_sc)
print("  Rodando métodos no GSE76225 (Biopsy)...")
raw_bp = run_all_methods(X_biopsy_sc)

# Anotação de perfis
prof_b  = {m: assign_profile(raw_b[m],  X_brushing_sc) for m in METHODS}
prof_bp = {m: assign_profile(raw_bp[m], X_biopsy_sc)   for m in METHODS}

# Distribuição por perfil
print("\n  Distribuição de perfis – K-means:")
for ds_name, prof_km in [("GSE76226 Brushing", prof_b["K-means"]),
                          ("GSE76225 Biopsy",   prof_bp["K-means"])]:
    dist = pd.Series(prof_km).value_counts()
    pct  = (dist / dist.sum() * 100).round(1)
    print(f"\n    {ds_name}:")
    for p in PROFILES:
        print(f"      {p:25s}: {dist.get(p,0):3d} ({pct.get(p,0):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  VALIDAÇÃO INTERNA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VALIDAÇÃO INTERNA")
print("=" * 70)

met_b  = compute_internal_metrics(X_brushing_sc, raw_b)
met_bp = compute_internal_metrics(X_biopsy_sc,   raw_bp)

print("\n  GSE76226 (Brushing):")
print(met_b.round(4).to_string())
print("\n  GSE76225 (Biopsy):")
print(met_bp.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 10.  ADJUSTED RAND INDEX
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ADJUSTED RAND INDEX (concordância entre métodos)")
print("=" * 70)

ari_b  = ari_matrix(raw_b,  METHODS)
ari_bp = ari_matrix(raw_bp, METHODS)

print("\n  GSE76226 (Brushing) – ARI:\n", ari_b.round(3).to_string())
print("\n  GSE76225 (Biopsy)   – ARI:\n", ari_bp.round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 11.  PERFIL DE CONSENSO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PERFIL DE CONSENSO (maioria entre 5 métodos)")
print("=" * 70)

cons_b,  agree_b  = build_consensus(prof_b,  METHODS)
cons_bp, agree_bp = build_consensus(prof_bp, METHODS)

full_cons_b  = agree_b  == 1.0
full_cons_bp = agree_bp == 1.0

print(f"\n  GSE76226 – consenso 5/5: {full_cons_b.sum()} / {len(full_cons_b)}")
print(f"  GSE76225 – consenso 5/5: {full_cons_bp.sum()} / {len(full_cons_bp)}")

# Consenso restrito (3 métodos geométricos)
GEO_METHODS = ["K-means", "Hierarchical", "K-medoids"]
cons_b3,  agree_b3  = build_consensus(prof_b,  GEO_METHODS)
cons_bp3, agree_bp3 = build_consensus(prof_bp, GEO_METHODS)

full_cons_b3  = agree_b3  == 1.0
full_cons_bp3 = agree_bp3 == 1.0

print(f"\n  GSE76226 – consenso 3/3 (geométricos): {full_cons_b3.sum()} / {len(full_cons_b3)}")
print(f"  GSE76225 – consenso 3/3 (geométricos): {full_cons_bp3.sum()} / {len(full_cons_bp3)}")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  CONCORDÂNCIA DE PERFIL NAS AMOSTRAS PAREADAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CONCORDÂNCIA DE PERFIL NAS AMOSTRAS PAREADAS (n=91)")
print("=" * 70)

for method in METHODS:
    pb_paired  = prof_b[method]           # brushing, todos 91
    pbp_paired = prof_bp[method][:N_PAIRED]  # biopsy, primeiros 91
    agree_pairs = (pb_paired == pbp_paired).sum()
    ari_pair = adjusted_rand_score(
        [PROFILES.index(x) for x in pb_paired],
        [PROFILES.index(x) for x in pbp_paired]
    )
    pct = agree_pairs / N_PAIRED * 100
    print(f"  {method:20s}: {agree_pairs}/{N_PAIRED} concordantes ({pct:.1f}%)  ARI={ari_pair:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 13.  TABELAS FINAIS DE CLASSIFICAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CONSTRUINDO TABELAS DE CLASSIFICAÇÃO")
print("=" * 70)

# ── 13a. GSE76226 (Brushing) ──────────────────────────────────────────────
table_brushing = pd.DataFrame(index=X_brushing.index)
table_brushing.index.name = "ID_Brushing_GSE76226"

# Expressão bruta
for g in GENES:
    table_brushing[f"expr_{g}"] = X_brushing[g]

# Scores de assinatura
table_brushing["T2_score_3GM"]  = X_brushing_sc[T2_GENES].mean(axis=1).values
table_brushing["T17_score_5GM"] = X_brushing_sc[T17_GENES].mean(axis=1).values

# Cluster de cada método
for m in METHODS:
    col = "cluster_" + m.lower().replace(" ", "_").replace("-", "_")
    table_brushing[col] = prof_b[m]

# Consenso 5/5
table_brushing["consensus_5_of_5"]       = cons_b
table_brushing["consensus_agreement_5"]  = agree_b.round(3)

# Consenso 3/3 (geométricos)
table_brushing["consensus_3geo"]          = cons_b3
table_brushing["consensus_agreement_3geo"]= agree_b3.round(3)

# Par biopsy
table_brushing["ID_Biopsy_pair_GSE76225"] = [
    X_biopsy.index[i] for i in range(N_PAIRED)
]

# ── 13b. GSE76225 (Biopsy) ───────────────────────────────────────────────
table_biopsy = pd.DataFrame(index=X_biopsy.index)
table_biopsy.index.name = "ID_Biopsy_GSE76225"

for g in GENES:
    table_biopsy[f"expr_{g}"] = X_biopsy[g]

table_biopsy["T2_score_3GM"]  = X_biopsy_sc[T2_GENES].mean(axis=1).values
table_biopsy["T17_score_5GM"] = X_biopsy_sc[T17_GENES].mean(axis=1).values

for m in METHODS:
    col = "cluster_" + m.lower().replace(" ", "_").replace("-", "_")
    table_biopsy[col] = prof_bp[m]

table_biopsy["consensus_5_of_5"]        = cons_bp
table_biopsy["consensus_agreement_5"]   = agree_bp.round(3)
table_biopsy["consensus_3geo"]          = cons_bp3
table_biopsy["consensus_agreement_3geo"]= agree_bp3.round(3)

table_biopsy["ID_Brushing_pair_GSE76226"] = (
    [X_brushing.index[i] for i in range(N_PAIRED)]
    + [None] * N_UNPAIRED
)
table_biopsy["Paired"] = ([True] * N_PAIRED + [False] * N_UNPAIRED)

# Salvar CSVs
out_brushing = f"{OUTDIR}GSE76226_brushing_cluster_assignments.csv"
out_biopsy   = f"{OUTDIR}GSE76225_biopsy_cluster_assignments.csv"
table_brushing.to_csv(out_brushing)
table_biopsy.to_csv(out_biopsy)

print(f"\n  Salvo: {out_brushing}")
print(f"  Salvo: {out_biopsy}")

print("\n  Prévia – GSE76226 Brushing (primeiras 5 linhas):")
preview_cols = ["T2_score_3GM", "T17_score_5GM",
                "cluster_k_means", "cluster_hierarchical",
                "consensus_5_of_5", "ID_Biopsy_pair_GSE76225"]
print(table_brushing[preview_cols].head().to_string())

print("\n  Prévia – GSE76225 Biopsy (primeiras 5 linhas):")
preview_cols2 = ["T2_score_3GM", "T17_score_5GM",
                 "cluster_k_means", "cluster_hierarchical",
                 "consensus_5_of_5", "Paired"]
print(table_biopsy[preview_cols2].head().to_string())

# ── 13c. Tabela de pares com ambos os clusters ────────────────────────────
pair_cluster = pair_table.copy()
for m in METHODS:
    col = "cluster_" + m.lower().replace(" ", "_").replace("-", "_")
    # Brushing
    mapping_b  = dict(zip(X_brushing.index, prof_b[m]))
    mapping_bp = dict(zip(X_biopsy.index,   prof_bp[m]))
    pair_cluster[f"brushing_{col}"] = pair_cluster["ID_Brushing_GSE76226"].map(mapping_b)
    pair_cluster[f"biopsy_{col}"]   = pair_cluster["ID_Biopsy_GSE76225"].map(mapping_bp)

pair_cluster["brushing_consensus"] = cons_b.tolist() + [None] * N_UNPAIRED
pair_cluster["biopsy_consensus"]   = list(cons_bp)

out_pairs = f"{OUTDIR}paired_samples_cluster_comparison.csv"
pair_cluster.to_csv(out_pairs, index=False)
print(f"\n  Salvo tabela de pares: {out_pairs}")

# ─────────────────────────────────────────────────────────────────────────────
# 14.  GERAÇÃO DE FIGURAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GERANDO FIGURAS")
print("=" * 70)

# Custom colormap para heatmap
CMAP_BIO = LinearSegmentedColormap.from_list(
    "bio", ["#000080", "#FFFFFF", "#FF0000"]
)

# ── Fig 1: Distribuições ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(
    "Fig 1 – Distribuições de Expressão Gênica: GSE76226 (Brushing) vs GSE76225 (Biopsy)",
    fontsize=13, fontweight="bold"
)
for i, g in enumerate(GENES):
    ax = axes[i // 4, i % 4]
    ax.hist(X_biopsy[g],   bins=20, alpha=0.6, color="#c0392b",
            label="Biopsy (GSE76225)",   density=True)
    ax.hist(X_brushing[g], bins=20, alpha=0.6, color="#2471a3",
            label="Brushing (GSE76226)", density=True)
    stat, pval = ks_2samp(X_biopsy[g], X_brushing[g])
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ax.set_title(f"$\\it{{{g}}}$\nKS={stat:.3f} {sig}", fontsize=10)
    ax.set_xlabel("Expressão"); ax.set_ylabel("Densidade")
    if i == 0:
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig1_distributions.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 1 salva.")

# ── Fig 2: Correlação inter-dataset ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Fig 2 – Correlação Inter-Dataset: GSE76226 vs GSE76225", fontsize=13, fontweight="bold")

# 2a: scatter médias gênicas
ax = axes[0]
for g in GENES:
    ax.scatter(mean_brushing[g], mean_biopsy[g], s=80, zorder=5)
    ax.annotate(g, (mean_brushing[g], mean_biopsy[g]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=8, fontstyle="italic")
z = np.polyfit(mean_brushing, mean_biopsy, 1)
xr = np.linspace(mean_brushing.min(), mean_brushing.max(), 50)
ax.plot(xr, np.poly1d(z)(xr), "k--", alpha=0.5)
ax.set_xlabel("GSE76226 Média"); ax.set_ylabel("GSE76225 Média")
ax.set_title(f"Médias Gênicas\nPearson r={r_pear:.3f}, Spearman ρ={r_spear:.3f}")

# 2b: KS stats
ax = axes[1]
ks_vals = [ks_table[i]["KS stat"] for i in range(len(ks_table))]
ks_pv   = [ks_table[i]["p-valor"] for i in range(len(ks_table))]
colors_ks = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in ks_pv]
ax.bar(GENES, ks_vals, color=colors_ks, edgecolor="black", linewidth=0.5)
ax.axhline(0.3, color="orange", linestyle="--", linewidth=1.5, label="Threshold 0.30")
ax.set_xticklabels(GENES, rotation=40, ha="right", fontstyle="italic")
ax.set_ylabel("KS Statistic"); ax.set_title("Teste KS por Gene\n(vermelho = p<0.05)")
ax.legend()

# 2c: boxplot comparativo
ax = axes[2]
pos = np.arange(len(GENES))
bp1 = ax.boxplot([X_biopsy[g] for g in GENES], positions=pos - 0.2, widths=0.35,
                  patch_artist=True,
                  boxprops=dict(facecolor="#f1948a", alpha=0.7),
                  medianprops=dict(color="darkred", linewidth=2), showfliers=False)
bp2 = ax.boxplot([X_brushing[g] for g in GENES], positions=pos + 0.2, widths=0.35,
                  patch_artist=True,
                  boxprops=dict(facecolor="#7fb3d3", alpha=0.7),
                  medianprops=dict(color="darkblue", linewidth=2), showfliers=False)
ax.set_xticks(pos)
ax.set_xticklabels(GENES, rotation=40, ha="right", fontstyle="italic")
ax.set_ylabel("Expressão Bruta")
ax.set_title("Distribuição Comparativa")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
           ["Biopsy (GSE76225)", "Brushing (GSE76226)"])
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig2_interdataset_correlation.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 2 salva.")

# ── Fig 3: PCA (5 métodos × 2 datasets) ──────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Fig 3 – PCA: 5 Métodos de Clustering × 2 Datasets", fontsize=13, fontweight="bold")

for ds_i, (Xs, raw, prof, label) in enumerate([
        (X_brushing_sc, raw_b,  prof_b,  "GSE76226 Brushing (n=91)"),
        (X_biopsy_sc,   raw_bp, prof_bp, "GSE76225 Biopsy (n=99)"),
]):
    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(Xs)
    vr  = pca.explained_variance_ratio_
    for m_i, m in enumerate(METHODS):
        ax = axes[ds_i, m_i]
        for pr, col in PALETTE.items():
            mask = prof[m] == pr
            ax.scatter(Xp[mask, 0], Xp[mask, 1], c=col, s=25, alpha=0.8)
        sil = silhouette_score(Xs, raw[m])
        ax.set_title(f"{m}\nSil={sil:.3f}", fontsize=9)
        ax.set_xlabel(f"PC1 ({vr[0]*100:.1f}%)", fontsize=8)
        if m_i == 0:
            ax.set_ylabel(f"{label}\nPC2 ({vr[1]*100:.1f}%)", fontsize=9)
        else:
            ax.set_ylabel(f"PC2 ({vr[1]*100:.1f}%)", fontsize=8)

patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
fig.legend(handles=patches, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.04), fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig3_PCA_all_methods.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 3 salva.")

# ── Fig 4: Silhueta ───────────────────────────────────────────────────────
def draw_silhouette(ax, Xs, raw_labels, prof_labels, title):
    sil_vals = silhouette_samples(Xs, raw_labels)
    y_lower = 10
    for pr in PROFILES:
        mask = prof_labels == pr
        sv   = np.sort(sil_vals[mask])
        size = sv.shape[0]
        ax.fill_betweenx(np.arange(y_lower, y_lower + size), 0, sv,
                          facecolor=PALETTE[pr], alpha=0.8)
        ax.text(-0.05, y_lower + size / 2,
                pr.split(" / ")[0], ha="right", va="center", fontsize=7)
        y_lower += size + 5
    avg = silhouette_score(Xs, raw_labels)
    ax.axvline(avg, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{title}\nSil médio={avg:.3f}", fontsize=10)
    ax.set_xlabel("Coeficiente de Silhueta")
    ax.set_xlim([-0.3, 1.0])

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle("Fig 4 – Gráficos de Silhueta (K-means e Hierárquico)", fontsize=13, fontweight="bold")
draw_silhouette(axes[0], X_brushing_sc, raw_b["K-means"],     prof_b["K-means"],     "Brushing – K-means")
draw_silhouette(axes[1], X_brushing_sc, raw_b["Hierarchical"],prof_b["Hierarchical"],"Brushing – Hierárquico")
draw_silhouette(axes[2], X_biopsy_sc,   raw_bp["K-means"],    prof_bp["K-means"],    "Biopsy – K-means")
draw_silhouette(axes[3], X_biopsy_sc,   raw_bp["Hierarchical"],prof_bp["Hierarchical"],"Biopsy – Hierárquico")
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig4_silhouette.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 4 salva.")

# ── Fig 5: Heatmaps ───────────────────────────────────────────────────────
def make_heatmap(ax, Xs, prof_labels, title):
    df = Xs[GENE_ORDER_HEATMAP].copy()
    df["Profile"]   = prof_labels
    df["T2score"]   = df[T2_GENES].mean(axis=1)
    df["T17score"]  = df[T17_GENES].mean(axis=1)
    parts = []
    for pr in PROFILES:
        sub = df[df["Profile"] == pr].copy()
        if pr == "T2HIGH / T17LOW":
            sub = sub.sort_values("T2score", ascending=False)
        elif pr == "T2LOW / T17HIGH":
            sub = sub.sort_values("T17score", ascending=False)
        else:
            sub = sub.sort_values(["T2score", "T17score"], ascending=False)
        parts.append(sub)
    df_s = pd.concat(parts)
    mat  = df_s[GENE_ORDER_HEATMAP].T
    sns.heatmap(mat, cmap=CMAP_BIO, center=0, vmin=-3, vmax=3,
                xticklabels=False, yticklabels=True, ax=ax,
                cbar_kws={"label": "Z-score", "shrink": 0.6})
    ax.yaxis.tick_right()
    ax.set_yticklabels(
        [f"$\\it{{{g}}}$" for g in GENE_ORDER_HEATMAP],
        rotation=0, fontsize=10
    )
    ax.set_xlabel("Amostras")
    ax.set_title(title, fontsize=12, fontweight="bold")
    x0 = 0
    for pr in PROFILES:
        cnt = (df_s["Profile"] == pr).sum()
        ax.axvline(x0 + cnt, color="black", lw=2)
        ax.text(x0 + cnt / 2, -0.5, pr.replace(" / ", "\n"),
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.2"))
        x0 += cnt

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Fig 5 – Heatmaps de Expressão por Perfil Transcriptômico (K-means, k=3)",
             fontsize=13, fontweight="bold")
make_heatmap(axes[0], X_brushing_sc, prof_b["K-means"],
             f"GSE76226 – Epithelial Brushing (n={len(X_brushing)})")
make_heatmap(axes[1], X_biopsy_sc,   prof_bp["K-means"],
             f"GSE76225 – Bronchial Biopsy (n={len(X_biopsy)})")
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig5_heatmaps.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 5 salva.")

# ── Fig 6: ARI ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fig 6 – Adjusted Rand Index: Concordância entre Métodos",
             fontsize=13, fontweight="bold")
for ax, ari, title in zip(axes,
                           [ari_b, ari_bp],
                           ["GSE76226 Brushing", "GSE76225 Biopsy"]):
    sns.heatmap(ari, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title(title)
    ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(METHODS, rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig6_ARI_heatmaps.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 6 salva.")

# ── Fig 7: Métricas internas ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Fig 7 – Métricas de Validação Interna (todos os métodos, ambos datasets)",
             fontsize=13, fontweight="bold")

metric_keys   = ["Silhueta ↑", "Calinski-Harabász ↑", "Davies-Bouldin ↓"]
colors_m = ["#e74c3c", "#9b59b6", "#f39c12", "#1abc9c", "#3498db"]
for col, mkey in enumerate(metric_keys):
    for row, (met, ds) in enumerate([(met_b, "GSE76226 Brushing"),
                                      (met_bp, "GSE76225 Biopsy")]):
        ax   = axes[row, col]
        vals = met[mkey]
        bars = ax.bar(METHODS, vals, color=colors_m,
                       edgecolor="black", linewidth=0.5)
        best_idx = int(vals.argmin() if "Bouldin" in mkey else vals.argmax())
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)
        ax.set_title(f"{ds} – {mkey}", fontsize=10)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig7_internal_metrics.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 7 salva.")

# ── Fig 8: Elbow + Gap ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Fig 8 – Seleção de k: Cotovelo, Silhueta e Estatística Gap",
             fontsize=13, fontweight="bold")
for row, (Ks, iner, sil_k, gap, gstd, ds) in enumerate([
        (Ks_b,  iner_b,  sil_b,  gap_b,  gstd_b,  "GSE76226 Brushing"),
        (Ks_bp, iner_bp, sil_bp, gap_bp, gstd_bp, "GSE76225 Biopsy"),
]):
    axes[row, 0].plot(Ks, iner,  "b-o", linewidth=2, markersize=7)
    axes[row, 0].axvline(3, color="red", linestyle="--", label="k=3")
    axes[row, 0].set_xlabel("k"); axes[row, 0].set_ylabel("Inércia")
    axes[row, 0].set_title(f"{ds} – Cotovelo")
    axes[row, 0].legend(); axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].plot(Ks, sil_k, "g-s", linewidth=2, markersize=7)
    axes[row, 1].axvline(Ks[int(np.argmax(sil_k))], color="orange",
                          linestyle="--", label=f"Best k={Ks[int(np.argmax(sil_k))]}")
    axes[row, 1].axvline(3, color="red", linestyle=":", label="k=3")
    axes[row, 1].set_xlabel("k"); axes[row, 1].set_ylabel("Silhueta Média")
    axes[row, 1].set_title(f"{ds} – Silhueta vs k")
    axes[row, 1].legend(); axes[row, 1].grid(True, alpha=0.3)

    axes[row, 2].plot(Ks, gap, "r-^", linewidth=2, markersize=7)
    axes[row, 2].fill_between(Ks, gap - gstd, gap + gstd, alpha=0.2, color="red")
    axes[row, 2].axvline(3, color="blue", linestyle="--", label="k=3")
    axes[row, 2].set_xlabel("k"); axes[row, 2].set_ylabel("Gap Statistic")
    axes[row, 2].set_title(f"{ds} – Gap Statistic")
    axes[row, 2].legend(); axes[row, 2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig8_optimal_k.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 8 salva.")

# ── Fig 9: Consenso ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fig 9 – Mapa de Consenso e Taxa de Concordância",
             fontsize=13, fontweight="bold")
for ax, Xs, cons, agree, ds in [
        (axes[0], X_brushing_sc, cons_b,  agree_b,  "GSE76226 Brushing (n=91)"),
        (axes[1], X_biopsy_sc,   cons_bp, agree_bp, "GSE76225 Biopsy (n=99)"),
]:
    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(Xs)
    for pr, col in PALETTE.items():
        mask = cons == pr
        if mask.sum() > 0:
            ax.scatter(Xp[mask, 0], Xp[mask, 1], c=col,
                       alpha=np.clip(agree[mask], 0.3, 1.0),
                       s=40 + 40 * agree[mask],
                       edgecolors="black", linewidths=0.3, label=pr)
    full = agree == 1.0
    ax.scatter(Xp[full, 0], Xp[full, 1], s=120,
               edgecolors="gold", facecolors="none",
               linewidths=2, label="5/5 consenso")
    vr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({vr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({vr[1]*100:.1f}%)")
    ax.set_title(ds); ax.legend(fontsize=7)
    # Inset: histograma de concordância
    axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
    cnts  = [
        ((agree >= 0.55) & (agree < 0.75)).sum(),
        ((agree >= 0.75) & (agree < 0.95)).sum(),
        (agree >= 0.95).sum(),
    ]
    axins.bar(range(3), cnts,
              color=["#e74c3c", "#f39c12", "#27ae60"],
              edgecolor="black", linewidth=0.5)
    axins.set_xticks(range(3))
    axins.set_xticklabels(["3/5", "4/5", "5/5"], fontsize=7)
    axins.set_title("Concordância", fontsize=7)
    axins.set_ylabel("n", fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig9_consensus.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 9 salva.")

# ── Fig 10: Scores T2 vs T17 ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Fig 10 – Scores de Assinatura T2 e T17 por Perfil Transcriptômico",
             fontsize=13, fontweight="bold")
for row, (Xs, prof_km, ds) in enumerate([
        (X_brushing_sc, prof_b["K-means"],  "GSE76226 Brushing"),
        (X_biopsy_sc,   prof_bp["K-means"], "GSE76225 Biopsy"),
]):
    ax = axes[row, 0]
    for pr, col in PALETTE.items():
        mask = prof_km == pr
        ax.hist(Xs[T2_GENES][mask].mean(axis=1), bins=15,
                alpha=0.6, color=col,
                label=pr.replace(" / ", "\n"), density=True)
    ax.set_xlabel("T2 mean z-score")
    ax.set_title(f"{ds}\nDistribuição T2")
    if row == 0: ax.legend(fontsize=6)

    ax = axes[row, 1]
    for pr, col in PALETTE.items():
        mask = prof_km == pr
        ax.hist(Xs[T17_GENES][mask].mean(axis=1), bins=15,
                alpha=0.6, color=col, density=True)
    ax.set_xlabel("T17 mean z-score")
    ax.set_title(f"{ds}\nDistribuição T17")

    ax = axes[row, 2]
    data_v = [Xs[T2_GENES][prof_km == pr].mean(axis=1).values
              for pr in PROFILES]
    vp = ax.violinplot(data_v, positions=range(3), showmedians=True)
    for pc, col in zip(vp["bodies"], PALETTE.values()):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([p.split(" / ")[0] for p in PROFILES],
                        rotation=20, fontsize=8)
    ax.set_ylabel("T2 Score")
    ax.set_title(f"{ds}\nViolin T2")

    ax = axes[row, 3]
    for pr, col in PALETTE.items():
        mask = prof_km == pr
        ax.scatter(Xs[T2_GENES][mask].mean(axis=1),
                   Xs[T17_GENES][mask].mean(axis=1),
                   c=col, alpha=0.7, s=30,
                   label=pr.replace(" / ", "\n"))
    ax.set_xlabel("T2 Score"); ax.set_ylabel("T17 Score")
    ax.set_title(f"{ds}\nT2 vs T17")
    ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
    if row == 0: ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig10_signature_scores.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 10 salva.")

# ── Fig 11: Concordância de perfil nas amostras PAREADAS ─────────────────
concordance_data = []
for method in METHODS:
    pb  = prof_b[method]           # brushing (91)
    pbp = prof_bp[method][:N_PAIRED]  # biopsy pareado (91)
    agree_pairs = (pb == pbp).sum()
    concordance_data.append({
        "Método"    : method,
        "Concordantes": agree_pairs,
        "Pct"       : agree_pairs / N_PAIRED * 100,
        "ARI"       : adjusted_rand_score(
            [PROFILES.index(x) for x in pb],
            [PROFILES.index(x) for x in pbp]
        ),
    })
conc_df = pd.DataFrame(concordance_data)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    f"Fig 11 – Concordância de Perfil nas Amostras Pareadas (n={N_PAIRED})\n"
    "Brushing (GSE76226) vs Biopsy (GSE76225)",
    fontsize=13, fontweight="bold"
)

# 11a: % concordância por método
ax = axes[0]
colors_bar = ["#c0392b" if p > 60 else "#e67e22" if p > 40 else "#95a5a6"
              for p in conc_df["Pct"]]
bars = ax.bar(conc_df["Método"], conc_df["Pct"], color=colors_bar,
               edgecolor="black", linewidth=0.5)
ax.axhline(100 / K, color="gray", linestyle="--", alpha=0.5, label="Acaso (33%)")
ax.set_ylabel("% Amostras Concordantes")
ax.set_title("Concordância de Perfil por Método")
ax.set_xticklabels(conc_df["Método"], rotation=30, ha="right")
ax.legend()
for bar, val in zip(bars, conc_df["Pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9)

# 11b: ARI por método
ax = axes[1]
ax.bar(conc_df["Método"], conc_df["ARI"], color="#3498db",
       edgecolor="black", linewidth=0.5)
ax.set_ylabel("ARI")
ax.set_title("Adjusted Rand Index (Brushing vs Biopsy)")
ax.set_xticklabels(conc_df["Método"], rotation=30, ha="right")
for i, (_, row) in enumerate(conc_df.iterrows()):
    ax.text(i, row["ARI"] + 0.01, f"{row['ARI']:.3f}",
            ha="center", va="bottom", fontsize=9)

# 11c: sankey-like confusion matrix (K-means, paired)
ax = axes[2]
pb_km  = prof_b["K-means"]
pbp_km = prof_bp["K-means"][:N_PAIRED]
conf   = pd.crosstab(
    pd.Series(pb_km,  name="Brushing"),
    pd.Series(pbp_km, name="Biopsy"),
    dropna=False
)
# Reindex to have all profiles
conf = conf.reindex(index=PROFILES, columns=PROFILES, fill_value=0)
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.7})
ax.set_title("K-means: Brushing vs Biopsy (pareados)")
ax.set_xlabel("Perfil Biopsy"); ax.set_ylabel("Perfil Brushing")
short = [p.split(" / ")[0] for p in PROFILES]
ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
ax.set_yticklabels(short, rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTDIR}Fig11_paired_concordance.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Fig 11 salva.")

# ─────────────────────────────────────────────────────────────────────────────
# 15.  RESUMO FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMO FINAL")
print("=" * 70)

best_method_b  = met_b["Silhueta ↑"].idxmax()
best_method_bp = met_bp["Silhueta ↑"].idxmax()

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  MÉTODO RECOMENDADO: K-means (k=3)                          │
  │  Validação obrigatória: Hierárquico Ward                    │
  │  Consenso robusto: 3 métodos geométricos (KM+HC+KMedoids)  │
  ├─────────────────────────────────────────────────────────────┤
  │  GSE76226 Brushing  – melhor método: {best_method_b:<20s}│
  │  GSE76225 Biopsy    – melhor método: {best_method_bp:<20s}│
  ├─────────────────────────────────────────────────────────────┤
  │  Amostras pareadas: {N_PAIRED}                                        │
  │  Amostras biopsy-only (sem par): {N_UNPAIRED}                          │
  │  Correlação Pearson (médias): r = {r_pear:.3f}                   │
  └─────────────────────────────────────────────────────────────┘
""")

print("  ARQUIVOS GERADOS:")
for fname in [
    "Fig1_distributions.png",
    "Fig2_interdataset_correlation.png",
    "Fig3_PCA_all_methods.png",
    "Fig4_silhouette.png",
    "Fig5_heatmaps.png",
    "Fig6_ARI_heatmaps.png",
    "Fig7_internal_metrics.png",
    "Fig8_optimal_k.png",
    "Fig9_consensus.png",
    "Fig10_signature_scores.png",
    "Fig11_paired_concordance.png",
    "GSE76226_brushing_cluster_assignments.csv",
    "GSE76225_biopsy_cluster_assignments.csv",
    "paired_samples_cluster_comparison.csv",
]:
    full = f"{OUTDIR}{fname}"
    exists = "✓" if os.path.exists(full) else "✗"
    print(f"    {exists}  {fname}")

print("\n  Análise concluída com sucesso.")
