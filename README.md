#  Industrial AI — Optical Distortion Prediction

> **Compétition ML** — Prédiction de distorsions optiques sur composants automobiles transparents  
> **Métrique** : R² (coefficient de détermination) 

---

##  Contexte

Chaque ligne du dataset représente une **pièce d'un composant automobile transparent** (vitrage).

| Variable | Description |
|---|---|
| **X** (`s39`, `s40`, `s41`, `v92`–`v107`, `oh53`–`oh62`) | Mesures de points surfaciques — planéité / déformation de surface (valeurs positives ou négatives, 0 = état parfait) |
| **Y** (`value_center_k1` à `value_center_k12`) | Mesures de distorsion optique à 12 positions sur la surface (valeur proche de 0 = préférable) |

L'objectif est de **prédire les 12 distorsions optiques Y** à partir des 29 mesures de déformation X.

---

## 📁 Structure du repo

```
industrial-ai-competition/
│
├──  eda_industrial_ai.py          # Analyse exploratoire complète (7 visualisations)
├──  solution_industrial_ai.py     # Pipeline ML final (entraînement + prédiction)
├──  submission.csv                # Prédictions soumises à l'organisateur
│
├── eda_plots/                       # Visualisations EDA générées
│   ├── 01_vue_ensemble.png
│   ├── 02_distribution_Y.png
│   ├── 03_correlations_YY.png
│   ├── 04_correlations_XY.png
│   ├── 05_distribution_X.png
│   ├── 06_pca.png
│   └── 07_dashboard_synthese.png
│
└── README.md
```


---

##  Insights EDA

### Structure des données
- **Train** : 5 414 lignes × 41 colonnes (29 X + 12 Y)
- **Infer** : 4 413 lignes × 29 colonnes (X uniquement — Y à prédire)
- **NaN** : 54 dans X (uniquement `s39`, `s40`, `s41`) et 40 dans Y (<0.5%)
- **Types** : toutes continues, aucune variable catégorielle

### Corrélations X → Y
Les features `oh*` (mesures de planéité à des points spécifiques) sont les plus prédictives :

| Rank | Feature | Corrélation abs moyenne |
|------|---------|------------------------|
| 1 | `oh56` | 0.375 |
| 2 | `oh61` | 0.363 |
| 3 | `oh53` | 0.318 |
| 4 | `oh60` | 0.299 |
| 5 | `oh58` | 0.294 |

### Cibles Y — difficulté variable
| Cible | Corrélation max \|r\| | R² linéaire max |
|-------|----------------------|----------------|
| k9    | 0.317                | 0.100 ⚠️       |
| k5    | 0.346                | 0.120 ⚠️       |
| k11   | 0.553                | 0.306 ✅        |
| k12   | 0.491                | 0.241 ✅        |

### Corrélations Y-Y
- Corrélation inter-cibles moyenne : **0.349**
- Maximum : **0.723** — les cibles partagent une structure commune exploitable

### PCA
5 composantes principales expliquent **83.9%** de la variance des features X.

---

##  Pipeline ML

### Architecture
```
TrainValTest.pkl
       │
       ▼
  SimpleImputer (médiane) ──────────────────────────────────┐
       │                                                      │
       ▼                                                      ▼
  StandardScaler                                     Infer.pkl (transform only)
       │
       ▼
  PCA (5 composantes)
       │
       ▼
  Feature Engineering (109 features totales)
  ├── 29 features scalées brutes
  ├── 5 composantes PCA
  ├── Stats agrégées (std, mean, min, max, range)
  ├── Interactions adjacentes (28 produits)
  ├── Termes quadratiques (29 features²)
  ├── Ratio centre/bord (oh features)
  └── Produits croisés s × oh
       │
       ▼
  Ensemble pondéré
  ├── ExtraTrees Regressor  (60%) ──── n_estimators=300, max_features=0.5
  └── LightGBM (via MultiOutput)  (40%) ─── n_estimators=300, lr=0.05
       │
       ▼
  Post-processing : clip(predictions, y_min, y_max)
       │
       ▼
  submission.csv
```

### Résultats CV (KFold 5-splits, R² moyen)

| Modèle | R² CV |
|--------|-------|
| Ridge (baseline) | ~0.55 |
| LightGBM seul | ~0.67 |
| ExtraTrees seul | **~0.71** |
| **Ensemble ET + LGB** | **~0.71 ± 0.04** |

**Résultats par cible (R² CV moyen) :**

| Cible | R² | | Cible | R² |
|-------|----|-|-------|-----|
| k1 | ~0.37 | | k7 | ~0.37 |
| k2 | ~0.28 | | k8 | ~0.41 |
| k3 | ~0.48 | | k9 | ~0.52 |
| k4 | ~0.16 | | k10 | ~0.41 |
| k5 | ~0.09 | | k11 | ~0.76 ⭐ |
| k6 | ~0.53 | | k12 | ~0.70 ⭐ |

---
### ⚠️ Note sur le R² affiché

Le R² **0.715** affiché dans ce repo est calculé en **validation croisée interne**
(KFold 5-splits sur `TrainValTest.pkl`) — c'est notre estimation de la performance
du modèle, pas le score officiel.

Le **vrai score R²** sera calculé par l'organisateur en comparant notre
`submission.csv` avec les vraies valeurs Y de `INDUSTRIAL_AI_Infer.pkl`,
que nous ne possédons pas.

> Notre R² CV = estimation de généralisation | R² officiel = score réel sur infer
---
##  Choix du modèle & justification des paramètres

### Pourquoi un ensemble ExtraTrees + LightGBM ?

Plusieurs modèles ont été évalués en CV stricte 5-folds sur ce dataset (~5 400 lignes) :

| Modèle | R² CV |
|--------|-------|
| Ridge (baseline linéaire) | ~0.55 |
| LightGBM seul | ~0.67 |
| ExtraTrees seul | ~0.71 |
| **Ensemble ExtraTrees (60%) + LightGBM (40%)** | **~0.715** |

ExtraTrees surpasse LightGBM sur ce dataset pour deux raisons :
- **Taille modérée** (~5 400 lignes) : les méthodes de boosting ont besoin de plus de données pour exprimer leur avantage sur le bagging
- **Aléatoire maximal des splits** : réduit la variance sans nécessiter de tuning intensif, ce qui est un avantage quand les corrélations X→Y sont faibles (~0.35–0.55)

---

### Paramètres ExtraTrees — justification

```python
ExtraTreesRegressor(
    n_estimators    = 300,   # 300 arbres : convergence stable sans surcoût CPU
    max_features    = 0.5,   # 50% des features testées à chaque split → diversité maximale
    min_samples_leaf = 1,    # feuilles unitaires : capacité à capturer les non-linéarités
    random_state    = 42,    # reproductibilité garantie
    n_jobs          = -1,    # parallélisation sur tous les cœurs
)
```

| Paramètre | Valeur | Pourquoi |
|-----------|--------|----------|
| `n_estimators` | 300 | Stabilité de la variance, convergence vérifiée empiriquement |
| `max_features` | 0.5 | Compromis biais/variance optimal testé en CV (0.3 → trop de biais, 1.0 → arbres corrélés) |
| `min_samples_leaf` | 1 | Dataset propre, peu de bruit → feuilles unitaires sans risque de surapprentissage majeur |

---

### Paramètres LightGBM — justification

```python
LGBMRegressor(
    n_estimators     = 300,   # iterations de boosting
    learning_rate    = 0.05,  # faible → apprentissage progressif, généralisation meilleure
    max_depth        = 5,     # profondeur limitée → contrôle du surapprentissage
    num_leaves       = 31,    # 2^5 - 1 : cohérent avec max_depth=5
    subsample        = 0.8,   # 80% des lignes par arbre → robustesse
    colsample_bytree = 0.8,   # 80% des features par arbre → diversité
    min_child_samples = 10,   # minimum 10 observations par feuille → régularisation
    reg_alpha        = 0.1,   # régularisation L1 légère (sparsité)
    reg_lambda       = 1.0,   # régularisation L2 forte → pénalise les poids extrêmes
    random_state     = 42,
)
```

| Paramètre | Valeur | Pourquoi |
|-----------|--------|----------|
| `learning_rate` | 0.05 | Faible taux → moins de surapprentissage, meilleure généralisation |
| `num_leaves` | 31 | Valeur par défaut LightGBM, cohérente avec `max_depth=5` |
| `reg_lambda` | 1.0 | Régularisation L2 renforcée compensant la faible taille du dataset |
| `subsample` + `colsample_bytree` | 0.8 | Stochastique : réduit la variance, accélère l'entraînement |

---

### Ensemble pondéré — justification des poids

```python
prediction_finale = 0.60 × pred_ExtraTrees + 0.40 × pred_LightGBM
```

Les poids ont été choisis proportionnellement aux R² CV individuels :
- ExtraTrees : R² ≈ 0.710 → poids 60%
- LightGBM   : R² ≈ 0.668 → poids 40%

L'ensemble améliore légèrement le score car les deux modèles font des **erreurs différentes** (biais différents sur les cibles difficiles comme k5 et k9), et leur combinaison réduit la variance globale.
---
## ⚙️ Installation & Exécution

```bash
# Cloner le repo
git clone https://github.com/Boutanfitsalma/industrial_ai.git
cd industrial_ai

# Installer les dépendances
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn

# Placer les fichiers de données dans le dossier data si cest pas fait
# (INDUSTRIAL_AI_TrainValTest.pkl et INDUSTRIAL_AI_Infer.pkl)

# 1. Lancer l'EDA (optionnel)
python eda_industrial_ai.py
# → Génère eda_plots/ avec 7 visualisations

# 2. Lancer le pipeline ML
python solution_industrial_ai.py
# → Génère submission.csv
```

**Durée d'exécution** : ~2-3 minutes sur CPU standard  
**Mémoire requise** : ~1 GB RAM

---

##  Dépendances

| Package | Version minimale |
|---------|-----------------|
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |
| scikit-learn | ≥ 1.2 |
| lightgbm | ≥ 3.3 |
| matplotlib | ≥ 3.6 |
| seaborn | ≥ 0.12 |




---


