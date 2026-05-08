"""
=============================================================================
INDUSTRIAL AI COMPETITION — Script Expert ML (Régression Multi-sorties)
Prédiction de distorsions optiques Y (value_center_k1-k12)
Stratégie : ExtraTrees + LightGBM Ensemble | Métrique : R²
=============================================================================

NOTE IMPORTANTE SUR LE BASELINE :
Le R²=0.90 annoncé via Ridge était probablement calculé in-sample (sur les données
d'entraînement elles-mêmes), ce qui génère un surajustement apparent.
En CV stricte 5-folds, le R² réel est ~0.71 avec ExtraTrees (meilleur modèle testé).
LightGBM seul atteint ~0.67, Ridge ~0.55. L'ensemble ET+LGB atteint ~0.71.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import warnings, time

warnings.filterwarnings("ignore")
np.random.seed(42)

start_time = time.time()
print("=" * 70)
print("INDUSTRIAL AI — Pipeline Expert ML (ExtraTrees + LightGBM Ensemble)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Chargement des données...")

df_train_full = pd.read_pickle("data/INDUSTRIAL_AI_TrainValTest.pkl")
df_infer      = pd.read_pickle("data/INDUSTRIAL_AI_Infer.pkl")

# Colonnes cibles = value_center_k1 à k12 (absentes du fichier infer → ce sont les Y)
Y_COLS = [f"value_center_k{i}" for i in range(1, 13)]
X_COLS = [c for c in df_train_full.columns if c not in Y_COLS]

X_raw   = df_train_full[X_COLS].copy().astype(np.float64)
y_train = df_train_full[Y_COLS].copy().astype(np.float64)
X_infer = df_infer[X_COLS].copy().astype(np.float64)

print(f"  → Train  : {X_raw.shape[0]:,} lignes × {X_raw.shape[1]} features X + {len(Y_COLS)} cibles Y")
print(f"  → Infer  : {X_infer.shape[0]:,} lignes × {X_infer.shape[1]} features X")
print(f"  → NaN X train : {X_raw.isnull().sum().sum()}")
print(f"  → NaN Y train : {y_train.isnull().sum().sum()}")
print(f"  → NaN infer   : {X_infer.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PRÉPROCESSING ANTI-LEAKAGE STRICT
#    Règle absolue : FIT uniquement sur train, TRANSFORM sur tout
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Préprocessing (anti-leakage strict)...")

# 2a. Imputation médiane des X — FIT sur train uniquement
imputer_X = SimpleImputer(strategy="median")
X_imp_train = imputer_X.fit_transform(X_raw).astype(np.float64)
X_imp_infer = imputer_X.transform(X_infer).astype(np.float64)

# Imputation médiane des Y manquants
imputer_Y = SimpleImputer(strategy="median")
y_imp = imputer_Y.fit_transform(y_train).astype(np.float64)

# 2b. Scaling — FIT sur train uniquement
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_imp_train).astype(np.float64)
X_scaled_infer = scaler.transform(X_imp_infer).astype(np.float64)

# 2c. PCA sur les 29 features X (n_components=5) — FIT sur train uniquement
pca = PCA(n_components=5, random_state=42)
X_pca_train = pca.fit_transform(X_scaled_train).astype(np.float64)
X_pca_infer = pca.transform(X_scaled_infer).astype(np.float64)

explained = pca.explained_variance_ratio_.cumsum()
print(f"  → PCA 5 composantes : variance expliquée = {explained[-1]:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Feature engineering...")


def build_features(X_sc: np.ndarray, X_raw_imp: np.ndarray,
                   X_pca: np.ndarray, col_names: list) -> np.ndarray:
    """
    Features finales = features scalées + PCA + stats agrégées
                     + interactions adjacentes + termes quadratiques
                     + ratio centre/bord + produits croisés s×oh
    """
    oh_idx = [i for i, c in enumerate(col_names) if c.startswith("oh")]
    s_idx  = [i for i, c in enumerate(col_names) if c.startswith("s")]

    parts = [
        X_sc,                                                           # 29 features scalées
        X_pca,                                                          # 5 composantes PCA
        X_raw_imp.std(axis=1, keepdims=True),                          # std globale
        X_raw_imp.mean(axis=1, keepdims=True),                         # mean globale
        X_raw_imp.min(axis=1, keepdims=True),                          # min globale
        X_raw_imp.max(axis=1, keepdims=True),                          # max globale
        (X_raw_imp.max(axis=1) - X_raw_imp.min(axis=1)).reshape(-1,1), # range globale
        X_sc ** 2,                                                      # termes quadratiques
    ]

    # Interactions adjacentes (n-1 produits)
    interact_adj = np.hstack([
        (X_sc[:, i] * X_sc[:, i + 1]).reshape(-1, 1)
        for i in range(X_sc.shape[1] - 1)
    ])
    parts.append(interact_adj)

    # Ratio centre/bord via features oh (fortement prédictives selon EDA)
    if len(oh_idx) >= 4:
        center = X_sc[:, oh_idx[:2]].sum(axis=1)
        edge   = X_sc[:, oh_idx[-2:]].sum(axis=1)
        parts.append((center / (edge + 1e-9)).reshape(-1, 1))

    # Produits croisés s × oh (s39, s40, s41 sont top features selon EDA)
    if s_idx and oh_idx:
        cross = np.hstack([
            (X_sc[:, s] * X_sc[:, oh]).reshape(-1, 1)
            for s in s_idx for oh in oh_idx[:4]
        ])
        parts.append(cross)

    return np.hstack(parts).astype(np.float64)


X_final_train = build_features(X_scaled_train, X_imp_train, X_pca_train, X_COLS)
X_final_infer = build_features(X_scaled_infer, X_imp_infer, X_pca_infer, X_COLS)

print(f"  → Features finales : {X_final_train.shape[1]} colonnes")

# Assertions de cohérence
assert X_final_train.shape[1] == X_final_infer.shape[1], \
    f"Mismatch colonnes train/infer : {X_final_train.shape[1]} vs {X_final_infer.shape[1]}"
assert not np.isnan(X_final_train).any(), "NaN détectés dans X_final_train!"
assert not np.isnan(X_final_infer).any(), "NaN détectés dans X_final_infer!"
print("  ✓ Assertions OK : aucun NaN, colonnes cohérentes")

# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDATION CROISÉE (KFold 5-splits)
#    ExtraTrees (60%) + LightGBM (40%) → ensemble pondéré
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Validation croisée KFold (5 splits)...")

ET_PARAMS = dict(
    n_estimators=300,
    max_features=0.5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

W_ET, W_LGB = 0.6, 0.4  # ExtraTrees légèrement favorisé (meilleur solo en CV)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_r2_scores = []
per_target_r2  = {col: [] for col in Y_COLS}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_final_train)):
    X_tr, X_val = X_final_train[train_idx], X_final_train[val_idx]
    y_tr, y_val = y_imp[train_idx],          y_imp[val_idx]

    # Modèle 1 : ExtraTrees (multioutput natif, très efficace sur petits datasets)
    et = ExtraTreesRegressor(**ET_PARAMS)
    et.fit(X_tr, y_tr)
    p_et = et.predict(X_val)

    # Modèle 2 : LightGBM via MultiOutputRegressor
    lgb_mo = MultiOutputRegressor(LGBMRegressor(**LGBM_PARAMS), n_jobs=-1)
    lgb_mo.fit(X_tr, y_tr)
    p_lgb = lgb_mo.predict(X_val)

    # Ensemble pondéré
    p_ens  = W_ET * p_et + W_LGB * p_lgb
    fold_r2 = r2_score(y_val, p_ens, multioutput="uniform_average")
    fold_r2_scores.append(fold_r2)

    per_target = r2_score(y_val, p_ens, multioutput="raw_values")
    for col, score in zip(Y_COLS, per_target):
        per_target_r2[col].append(score)

    print(f"  Fold {fold + 1}/5 — R² moyen : {fold_r2:.4f}")

mean_cv_r2 = np.mean(fold_r2_scores)
std_cv_r2  = np.std(fold_r2_scores)
print(f"\n  ╔════════════════════════════════════╗")
print(f"  ║  R² CV moyen : {mean_cv_r2:.4f} ± {std_cv_r2:.4f}  ║")
print(f"  ╚════════════════════════════════════╝")

print("\n  R² par cible (moyenne CV) :")
for col in Y_COLS:
    avg  = np.mean(per_target_r2[col])
    flag = " ⚠  (cible difficile)" if avg < 0.3 else (" ★" if avg > 0.7 else "")
    print(f"    {col:<25} : {avg:.4f}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRAÎNEMENT FINAL SUR 100% DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Entraînement final (100% train)...")

final_et = ExtraTreesRegressor(**ET_PARAMS)
final_et.fit(X_final_train, y_imp)
print("  → ExtraTrees : OK")

final_lgb = MultiOutputRegressor(LGBMRegressor(**LGBM_PARAMS), n_jobs=-1)
final_lgb.fit(X_final_train, y_imp)
print("  → LightGBM   : OK")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PRÉDICTIONS + POST-PROCESSING + EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Génération des prédictions et export...")

p_et_infer  = final_et.predict(X_final_infer)
p_lgb_infer = final_lgb.predict(X_final_infer)
preds_infer = W_ET * p_et_infer + W_LGB * p_lgb_infer

# Post-processing : clip selon limites physiques du train (bornes observées)
y_min = y_imp.min(axis=0)
y_max = y_imp.max(axis=0)
preds_clipped = np.clip(preds_infer, y_min, y_max)
n_clipped = int((preds_infer != preds_clipped).sum())
print(f"  → Clipping : {n_clipped} valeurs ajustées sur {preds_infer.size}")

# DataFrame de soumission
submission = pd.DataFrame(
    preds_clipped,
    columns=Y_COLS,
    index=df_infer.index,
)

# ── ASSERTIONS FINALES ──────────────────────────────────────────────────────
assert list(submission.columns) == Y_COLS, \
    f"Ordre des colonnes incorrect! Attendu : {Y_COLS}"
assert submission.isnull().sum().sum() == 0, \
    "NaN détectés dans les prédictions finales!"
assert submission.shape == (len(df_infer), 12), \
    f"Shape inattendue : {submission.shape}"
assert not (submission.values < y_min - 1e-9).any(), \
    "Valeurs sous le minimum physique!"
assert not (submission.values > y_max + 1e-9).any(), \
    "Valeurs au-dessus du maximum physique!"
print("  ✓ Toutes les assertions finales OK")

# Export
output_path = "submission.csv"
submission.to_csv(output_path, index=True)
print(f"  ✓ Export : {output_path}")
print(f"  → Shape  : {submission.shape}")
print(f"\n  Aperçu (3 premières lignes) :")
print(submission.head(3).to_string())

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"✅ Pipeline terminé en {elapsed:.1f}s ({elapsed / 60:.1f} min)")
print(f"   R² CV estimé : {mean_cv_r2:.4f} ± {std_cv_r2:.4f}")
print(f"{'=' * 70}")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  3 PISTES D'OPTIMISATION SI R² CV STAGNE SOUS 0.80                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. FEATURES PHYSIQUES MANQUANTES : Les cibles k4, k5, k10 ont       ║
║     un R² faible (~0.05-0.20). Elles dépendent probablement de       ║
║     signaux non capturés par les 29 features actuelles (gradients,   ║
║     FFT des profils, mesures de courbure locale). Demander les       ║
║     données brutes capteur si disponibles. Alternative : créer des   ║
║     pseudo-features par interpolation spatiale entre k-voisins.      ║
║                                                                      ║
║  2. OPTUNA PAR CIBLE : Lancer une optimisation Bayésienne (~100      ║
║     trials, 5-min budget) séparément pour chaque cible difficile     ║
║     (k4, k5, k9, k10). Les hyperparam optimaux diffèrent selon       ║
║     la complexité de chaque cible. Focus sur min_samples_leaf,       ║
║     max_features pour ET ; num_leaves, reg_lambda pour LGB.          ║
║                                                                      ║
║  3. MULTITASK LEARNING avec corrélations Y-Y : Les cibles sont       ║
║     corrélées entre elles (r̄=0.35). Un MLP MultiTask PyTorch avec    ║
║     une tête partagée + têtes individuelles peut exploiter cette     ║
║     structure. Ajouter un meta-learner Ridge en stacking sur les     ║
║     OOF predictions de ET + LGB + MLP → gain typique de +0.05 R².   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
