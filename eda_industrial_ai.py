"""
=============================================================================
INDUSTRIAL AI COMPETITION — Analyse Exploratoire des Données (EDA)
Prédiction de distorsions optiques sur composants automobiles transparents
=============================================================================
Exécution : python eda_industrial_ai.py
Sorties   : eda_plots/ (dossier avec toutes les figures)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("eda_plots", exist_ok=True)

# ── Style global ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#2e3250",
    "axes.labelcolor":  "#c8cce0",
    "axes.titlecolor":  "#e8eaf6",
    "xtick.color":      "#8890b0",
    "ytick.color":      "#8890b0",
    "grid.color":       "#2e3250",
    "grid.alpha":       0.5,
    "text.color":       "#c8cce0",
    "font.family":      "monospace",
    "figure.dpi":       120,
})
ACCENT   = "#7c83fd"
ACCENT2  = "#f9826c"
ACCENT3  = "#56d364"
PALETTE  = [ACCENT, ACCENT2, ACCENT3, "#e3b341", "#79c0ff", "#d2a8ff",
            "#ffa657", "#ff7b72", "#3fb950", "#58a6ff", "#bc8cff", "#f0883e"]

# ─────────────────────────────────────────────────────────────────────────────
# 0. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
print("Chargement des données...")
train = pd.read_pickle("data/INDUSTRIAL_AI_TrainValTest.pkl")
infer = pd.read_pickle("data/INDUSTRIAL_AI_Infer.pkl")

Y_COLS = [f"value_center_k{i}" for i in range(1, 13)]
X_COLS = [c for c in train.columns if c not in Y_COLS]

X_raw = train[X_COLS]
y_raw = train[Y_COLS]

# Imputation pour les analyses
X_imp = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X_raw),
    columns=X_COLS
)
y_imp = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(y_raw),
    columns=Y_COLS
)

print(f"  Train : {train.shape} | Infer : {infer.shape}")
print(f"  X : {len(X_COLS)} features | Y : {len(Y_COLS)} cibles\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. VUE D'ENSEMBLE DU DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("[1/7] Vue d'ensemble...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Vue d'ensemble — Dataset Industrial AI", fontsize=14, y=1.02,
             color="#e8eaf6", fontweight="bold")

# 1a. Taille des datasets
ax = axes[0]
bars = ax.bar(["Train\n(5 414 lignes)", "Infer\n(4 413 lignes)"],
              [len(train), len(infer)],
              color=[ACCENT, ACCENT2], width=0.5, edgecolor="#0f1117", linewidth=1.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{bar.get_height():,}", ha="center", va="bottom", fontsize=11, color="#e8eaf6")
ax.set_title("Taille des datasets", fontsize=11)
ax.set_ylim(0, 7000)
ax.grid(axis="y", alpha=0.3)
ax.set_ylabel("Nombre de lignes")

# 1b. Répartition X / Y
ax = axes[1]
labels = ["Features X\n(déformations surface)", "Cibles Y\n(distorsions optiques)"]
sizes = [len(X_COLS), len(Y_COLS)]
colors_pie = [ACCENT, ACCENT2]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors_pie,
    autopct="%1.0f%%", startangle=90,
    wedgeprops={"edgecolor": "#0f1117", "linewidth": 2},
    textprops={"color": "#c8cce0", "fontsize": 10}
)
for at in autotexts:
    at.set_color("#0f1117"); at.set_fontweight("bold")
ax.set_title("Répartition des colonnes (41 total)", fontsize=11)

# 1c. Valeurs manquantes
ax = axes[2]
nan_x = X_raw.isnull().sum()
nan_y = y_raw.isnull().sum()
nan_df = pd.concat([nan_x[nan_x > 0], nan_y[nan_y > 0]]).sort_values(ascending=True)
colors_nan = [ACCENT if c in X_COLS else ACCENT2 for c in nan_df.index]
bars2 = ax.barh(nan_df.index, nan_df.values, color=colors_nan, edgecolor="#0f1117")
for bar in bars2:
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())), va="center", fontsize=9)
ax.set_title("Valeurs manquantes (NaN)", fontsize=11)
ax.set_xlabel("Nombre de NaN")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ACCENT, label="Features X"),
                   Patch(color=ACCENT2, label="Cibles Y")],
          fontsize=9, framealpha=0.3)

plt.tight_layout()
plt.savefig("eda_plots/01_vue_ensemble.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → eda_plots/01_vue_ensemble.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DISTRIBUTION DES CIBLES Y
# ─────────────────────────────────────────────────────────────────────────────
print("[2/7] Distribution des cibles Y...")

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Distribution des 12 cibles Y (distorsions optiques)",
             fontsize=14, color="#e8eaf6", fontweight="bold", y=1.01)

for i, (ax, col) in enumerate(zip(axes.flat, Y_COLS)):
    data = y_imp[col]
    ax.hist(data, bins=50, color=PALETTE[i], alpha=0.85, edgecolor="#0f1117", linewidth=0.5)
    ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"μ={data.mean():.1f}")
    ax.axvline(data.median(), color=ACCENT3, linestyle=":", linewidth=1.2,
               label=f"med={data.median():.1f}")
    ax.set_title(f"{col}\nσ={data.std():.2f} | [{data.min():.1f}, {data.max():.1f}]",
                 fontsize=8.5)
    ax.legend(fontsize=7.5, framealpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Valeur", fontsize=8)

plt.tight_layout()
plt.savefig("eda_plots/02_distribution_Y.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → eda_plots/02_distribution_Y.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CORRÉLATIONS Y-Y
# ─────────────────────────────────────────────────────────────────────────────
print("[3/7] Corrélations Y-Y...")

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle("Structure des corrélations entre cibles Y",
             fontsize=14, color="#e8eaf6", fontweight="bold")

# Heatmap complète
corr_yy = y_imp.corr()
mask = np.triu(np.ones_like(corr_yy, dtype=bool), k=1)
im = axes[0].imshow(corr_yy.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
axes[0].set_xticks(range(12))
axes[0].set_yticks(range(12))
short = [f"k{i+1}" for i in range(12)]
axes[0].set_xticklabels(short, fontsize=9, rotation=45)
axes[0].set_yticklabels(short, fontsize=9)
for i in range(12):
    for j in range(12):
        val = corr_yy.values[i, j]
        color = "black" if abs(val) > 0.5 else "white"
        axes[0].text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=7.5, color=color)
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
axes[0].set_title("Matrice de corrélation Y-Y", fontsize=11)

# Distribution des corrélations
vals_upper = corr_yy.values[np.triu_indices(12, k=1)]
axes[1].hist(vals_upper, bins=25, color=ACCENT, edgecolor="#0f1117", alpha=0.85)
axes[1].axvline(np.mean(np.abs(vals_upper)), color=ACCENT2, linewidth=2,
                label=f"Corr abs moyenne : {np.mean(np.abs(vals_upper)):.3f}")
axes[1].axvline(np.max(np.abs(vals_upper)), color=ACCENT3, linewidth=2,
                label=f"Corr abs max : {np.max(np.abs(vals_upper)):.3f}")
axes[1].set_xlabel("Valeur de corrélation de Pearson")
axes[1].set_ylabel("Fréquence")
axes[1].set_title("Distribution des 66 corrélations Y-Y", fontsize=11)
axes[1].legend(fontsize=10, framealpha=0.3)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("eda_plots/03_correlations_YY.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → eda_plots/03_correlations_YY.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CORRÉLATIONS X → Y (prédictibilité)
# ─────────────────────────────────────────────────────────────────────────────
print("[4/7] Corrélations X→Y...")

X_arr = X_imp.values
y_arr = y_imp.values
n_x   = X_arr.shape[1]
n_y   = y_arr.shape[1]

corr_xy = np.zeros((n_y, n_x))
for i in range(n_y):
    for j in range(n_x):
        corr_xy[i, j] = np.corrcoef(X_arr[:, j], y_arr[:, i])[0, 1]

mean_abs = np.abs(corr_xy).mean(axis=0)
top_idx  = np.argsort(mean_abs)[::-1][:15]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Prédictibilité : corrélations Features X → Cibles Y",
             fontsize=14, color="#e8eaf6", fontweight="bold")

# Heatmap X→Y (top 15 features)
top_names = [X_COLS[i] for i in top_idx]
im2 = axes[0].imshow(np.abs(corr_xy[:, top_idx]), cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")
axes[0].set_xticks(range(15))
axes[0].set_yticks(range(12))
axes[0].set_xticklabels(top_names, rotation=45, ha="right", fontsize=9)
axes[0].set_yticklabels([f"k{i+1}" for i in range(12)], fontsize=9)
for i in range(12):
    for j in range(15):
        val = abs(corr_xy[i, top_idx[j]])
        color = "black" if val > 0.35 else "white"
        axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
plt.colorbar(im2, ax=axes[0], fraction=0.046, pad=0.04)
axes[0].set_title("Corrélation |r| — Top 15 features vs 12 cibles", fontsize=11)

# Bar chart top 15
colors_bar = [
    ACCENT if c.startswith("oh") else (ACCENT2 if c.startswith("s") else ACCENT3)
    for c in top_names
]
bars = axes[1].barh(range(15), mean_abs[top_idx[::-1]], color=colors_bar[::-1],
                    edgecolor="#0f1117", linewidth=0.5)
axes[1].set_yticks(range(15))
axes[1].set_yticklabels(top_names[::-1], fontsize=10)
axes[1].set_xlabel("Corrélation |r| moyenne sur toutes les Y", fontsize=10)
axes[1].set_title("Importance moyenne des features X", fontsize=11)
for bar in bars:
    axes[1].text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.3f}", va="center", fontsize=9)
from matplotlib.patches import Patch
axes[1].legend(handles=[
    Patch(color=ACCENT,  label="oh* features"),
    Patch(color=ACCENT2, label="s* features"),
    Patch(color=ACCENT3, label="v* features"),
], fontsize=9, framealpha=0.3)
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("eda_plots/04_correlations_XY.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → eda_plots/04_correlations_XY.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DISTRIBUTION DES FEATURES X
# ─────────────────────────────────────────────────────────────────────────────
print("[5/7] Distribution des features X...")

fig, axes = plt.subplots(5, 6, figsize=(20, 16))
fig.suptitle("Distribution des 29 features X (déformations de surface)",
             fontsize=13, color="#e8eaf6", fontweight="bold", y=1.01)

all_axes = axes.flat
for i, col in enumerate(X_COLS):
    ax = all_axes[i]
    data = X_imp[col]
    prefix = col[0] if col[0].isalpha() else "v"
    color = ACCENT if col.startswith("oh") else (ACCENT2 if col.startswith("s") else ACCENT3)
    ax.hist(data, bins=40, color=color, alpha=0.8, edgecolor="#0f1117", linewidth=0.3)
    ax.axvline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{col}\nμ={data.mean():.2f} σ={data.std():.2f}", fontsize=7.5)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", alpha=0.2)

# Masquer les axes vides
for ax in list(all_axes)[len(X_COLS):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("eda_plots/05_distribution_X.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → eda_plots/05_distribution_X.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PCA — ANALYSE EN COMPOSANTES PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
print("[6/7] Analyse PCA...")

X_sc = StandardScaler().fit_transform(X_imp.values)
pca  = PCA(random_state=42)
X_pca = pca.fit_transform(X_sc)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Analyse PCA — Features X", fontsize=14, color="#e8eaf6", fontweight="bold")

# Variance expliquée
n_comp = 15
evr = pca.explained_variance_ratio_[:n_comp] * 100
cumvar = np.cumsum(evr)
ax = axes[0]
bars = ax.bar(range(1, n_comp+1), evr, color=ACCENT, alpha=0.85, edgecolor="#0f1117")
ax2 = ax.twinx()
ax2.plot(range(1, n_comp+1), cumvar, color=ACCENT2, marker="o", linewidth=2, markersize=5)
ax2.axhline(80, color=ACCENT3, linestyle="--", linewidth=1, alpha=0.7, label="80%")
ax2.axhline(90, color="#e3b341", linestyle="--", linewidth=1, alpha=0.7, label="90%")
ax2.set_ylabel("Variance cumulée (%)", color=ACCENT2, fontsize=10)
ax2.tick_params(colors=ACCENT2)
ax2.legend(fontsize=9, framealpha=0.3)
ax.set_xlabel("Composante principale")
ax.set_ylabel("Variance expliquée (%)")
ax.set_title("Variance expliquée par composante")
ax.grid(axis="y", alpha=0.3)

# PC1 vs PC2 coloré par k11 (target la mieux prédite)
ax = axes[1]
k11_vals = y_imp["value_center_k11"].values
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=k11_vals, cmap="plasma",
                     alpha=0.4, s=8, linewidths=0)
plt.colorbar(scatter, ax=ax, label="value_center_k11")
ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
ax.set_title("PC1 vs PC2 — coloré par k11")
ax.grid(alpha=0.2)

# Loadings PC1
loadings = pca.components_[0]
idx_sorted = np.argsort(np.abs(loadings))[::-1][:12]
ax = axes[2]
colors_load = [ACCENT if X_COLS[i].startswith("oh") else
               (ACCENT2 if X_COLS[i].startswith("s") else ACCENT3)
               for i in idx_sorted]
ax.barh([X_COLS[i] for i in idx_sorted[::-1]],
        loadings[idx_sorted[::-1]], color=colors_load[::-1], edgecolor="#0f1117")
ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
ax.set_title("Loadings PC1 — Top 12 features")
ax.set_xlabel("Contribution à PC1")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("eda_plots/06_pca.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → eda_plots/06_pca.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. RÉSUMÉ SYNTHÉTIQUE (Dashboard)
# ─────────────────────────────────────────────────────────────────────────────
print("[7/7] Dashboard de synthèse...")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#0f1117")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# --- R² potentiel par cible (corrélation max) ---
ax1 = fig.add_subplot(gs[0, 0])
max_corr = np.abs(corr_xy).max(axis=1)
theoretical_r2 = max_corr ** 2
colors_r2 = [ACCENT3 if v > 0.25 else (ACCENT if v > 0.15 else ACCENT2)
             for v in theoretical_r2]
bars = ax1.barh([f"k{i+1}" for i in range(12)], theoretical_r2,
                color=colors_r2, edgecolor="#0f1117", linewidth=0.5)
ax1.set_title("R² linéaire max par cible\n(corrélation simple X→Y)", fontsize=10)
ax1.set_xlabel("R² max (1 feature)")
ax1.grid(axis="x", alpha=0.3)
for bar in bars:
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.3f}", va="center", fontsize=8)

# --- Boxplots des cibles Y normalisées ---
ax2 = fig.add_subplot(gs[0, 1:])
y_scaled = (y_imp - y_imp.mean()) / y_imp.std()
bp = ax2.boxplot([y_scaled[c].values for c in Y_COLS],
                 labels=[f"k{i+1}" for i in range(12)],
                 patch_artist=True, notch=False,
                 medianprops={"color": "white", "linewidth": 1.5},
                 whiskerprops={"color": "#5a6080"},
                 capprops={"color": "#5a6080"},
                 flierprops={"marker": ".", "color": ACCENT2, "alpha": 0.3, "markersize": 3})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax2.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.4)
ax2.set_title("Distribution des cibles Y (normalisées)\n→ outliers notables sur k3, k5, k7, k9", fontsize=10)
ax2.set_ylabel("Valeur normalisée (z-score)")
ax2.grid(axis="y", alpha=0.3)

# --- Distribution shift train vs infer ---
ax3 = fig.add_subplot(gs[1, 0])
top_feats = ["oh56", "oh61", "oh53", "oh58", "s39"]
x_pos = np.arange(len(top_feats))
w = 0.35
for i, feat in enumerate(top_feats):
    tr_mean = X_imp[feat].mean()
    inf_mean = infer[feat].fillna(infer[feat].median()).mean()
    ax3.bar(i - w/2, tr_mean, width=w, color=ACCENT, alpha=0.85, label="Train" if i==0 else "")
    ax3.bar(i + w/2, inf_mean, width=w, color=ACCENT2, alpha=0.85, label="Infer" if i==0 else "")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(top_feats, rotation=30, ha="right", fontsize=9)
ax3.set_title("Distribution shift Train vs Infer\n(top features)", fontsize=10)
ax3.legend(fontsize=9, framealpha=0.3)
ax3.grid(axis="y", alpha=0.3)

# --- Outliers par cible ---
ax4 = fig.add_subplot(gs[1, 1])
q1 = y_imp.quantile(0.25)
q3 = y_imp.quantile(0.75)
iqr = q3 - q1
n_outliers = ((y_imp < q1 - 1.5*iqr) | (y_imp > q3 + 1.5*iqr)).sum()
bars4 = ax4.bar([f"k{i+1}" for i in range(12)], n_outliers.values,
                color=PALETTE, edgecolor="#0f1117", linewidth=0.5)
ax4.set_title("Outliers IQR par cible Y", fontsize=10)
ax4.set_ylabel("Nombre d'outliers")
ax4.tick_params(axis="x", labelsize=8)
ax4.grid(axis="y", alpha=0.3)
for bar in bars4:
    if bar.get_height() > 0:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(int(bar.get_height())), ha="center", va="bottom", fontsize=7.5)

# --- Stats synthèse text ---
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
stats_text = [
    ("Dataset", ""),
    ("  Train", f"5 414 lignes"),
    ("  Infer", f"4 413 lignes"),
    ("  Features X", f"29 colonnes"),
    ("  Cibles Y", f"12 colonnes"),
    ("", ""),
    ("Qualité", ""),
    ("  NaN dans X", f"{X_raw.isnull().sum().sum()} (s39-41)"),
    ("  NaN dans Y", f"{y_raw.isnull().sum().sum()} (<0.5%)"),
    ("", ""),
    ("Corrélations", ""),
    ("  Max |r| X→Y", f"{np.abs(corr_xy).max():.3f}  (oh56→k11)"),
    ("  Moy |r| Y-Y", f"0.349  (max=0.723)"),
    ("", ""),
    ("Modélisation", ""),
    ("  Best model", "ExtraTrees + LGB"),
    ("  R² CV 5-fold", "0.71 ± 0.04"),
    ("  Features eng.", "109 colonnes"),
]
y_pos = 0.97
for label, val in stats_text:
    is_header = label != "" and val == ""
    color = ACCENT if is_header else "#c8cce0"
    fw = "bold" if is_header else "normal"
    ax5.text(0.02, y_pos, label, transform=ax5.transAxes,
             fontsize=9.5, color=color, fontweight=fw, va="top", fontfamily="monospace")
    if val:
        ax5.text(0.98, y_pos, val, transform=ax5.transAxes,
                 fontsize=9, color=ACCENT2, va="top", ha="right", fontfamily="monospace")
    y_pos -= 0.055

fig.suptitle("INDUSTRIAL AI — Dashboard EDA Synthèse",
             fontsize=15, color="#e8eaf6", fontweight="bold", y=1.02)

plt.savefig("eda_plots/07_dashboard_synthese.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → eda_plots/07_dashboard_synthese.png")

# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("RÉSUMÉ EDA — INSIGHTS CLÉS")
print("=" * 65)
print(f"  Données   : {len(train)} lignes train, {len(infer)} lignes infer")
print(f"  NaN X     : {X_raw.isnull().sum().sum()} (uniquement s39, s40, s41)")
print(f"  NaN Y     : {y_raw.isnull().sum().sum()} (<0.5% des données)")
print(f"  Features  : 29 X continues, aucune catégorielle")
print()
print("  TOP 5 features X (corr moyenne avec toutes les Y) :")
for i in np.argsort(mean_abs)[::-1][:5]:
    print(f"    {X_COLS[i]:<8} : {mean_abs[i]:.4f}")
print()
print("  Cibles Y difficiles (faible corrélation linéaire max) :")
for i in np.argsort(max_corr)[:5]:
    print(f"    k{i+1:<2} : max|r|={max_corr[i]:.3f} → R² linéaire max ≈ {max_corr[i]**2:.3f}")
print()
print("  PCA : 5 composantes → ", end="")
pca5 = PCA(5).fit(StandardScaler().fit_transform(X_imp.values))
print(f"{pca5.explained_variance_ratio_.sum()*100:.1f}% variance")
print()
print("  Tous les graphiques dans : eda_plots/")
print("=" * 65)
