# DOLPH-X — Documentation technique

> Pipeline de prédiction du décrochage étudiant avec explicabilité (XAI)  
> Mémoire de Master — Intelligence Artificielle Explicable en contexte éducatif

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture du pipeline](#2-architecture-du-pipeline)
3. [Modèles retenus](#3-modèles-retenus)
4. [Sélection des instances (filtre de consensus)](#4-sélection-des-instances-filtre-de-consensus)
5. [Méthodes XAI implémentées](#5-méthodes-xai-implémentées)
6. [Ajustements techniques documentés](#6-ajustements-techniques-documentés)
7. [Architecture du code](#7-architecture-du-code)
8. [Limitations connues](#8-limitations-connues)
9. [Glossaire](#9-glossaire)

---

## 1. Vue d'ensemble du projet

DOLPH-X est un pipeline expérimental qui combine **prédiction du décrochage étudiant** et **explicabilité de l'IA (XAI)**. À partir d'un dataset de 128 étudiants, le système entraîne plusieurs modèles de machine learning, sélectionne les plus performants, puis génère des explications locales et contrefactuelles sur des instances représentatives.

L'objectif n'est pas uniquement de prédire — c'est de **rendre les prédictions interprétables** pour des utilisateurs non experts (enseignants, conseillers pédagogiques), dans le cadre d'une étude qualitative sur la perception des explications XAI.

### Profils étudiants

Le dataset distingue quatre profils selon deux axes :

| Profil | Signification | Effectif total |
|--------|--------------|---------------|
| **E+P+** | Engagé + Performant | 7 instances |
| **E+P−** | Engagé + Non performant | 4 instances |
| **E−P+** | Non engagé + Performant | 9 instances |
| **E−P−** | Non engagé + Non performant | 6 instances |

> ⚠️ Le profil **E+P−** est le moins représenté du dataset (14,8%). Cela constitue une limitation structurelle inhérente à la taille du jeu de données.

---

## 2. Architecture du pipeline

```
[Dataset brut — 128 étudiants]
         │
         ▼
[Preprocessing]
  • Suppression des variables constantes (ex: TOTAL_STUDENTS_NUMBER)
  • Normalisation via StandardScaler
  • Rééchantillonnage SMOTE (k_neighbors=3)
         │
         ▼
[Entraînement — 11 modèles ML]
         │
         ▼
[Sélection — 3 modèles retenus]
  Random Forest · XGBoost · Decision Tree
         │
         ▼
[Filtre de consensus RF + XGBoost]
  → 13 instances sélectionnées sur 26
         │
         ▼
[Génération des explications XAI]
  LIME · MC-LIME · SHAP · DiCE
         │
         ▼
[Évaluation — LLM-as-a-Judge (Phase 3)] U [Interview]
```

---

## 3. Modèles retenus

Sur les 11 modèles entraînés, **3 ont été retenus** pour les analyses XAI selon des critères de performance et de diversité méthodologique.

| Modèle | F1-Score | AUC-ROC | Type | Rôle |
|--------|----------|---------|------|------|
| **Random Forest** | 0.7997 | 0.961 | Black box (bagging) | Meilleur modèle global |
| **XGBoost** | 0.7987 | 0.960 | Black box (boosting) | Représentant famille boosting |
| **Decision Tree** | 0.7368 | 0.847 | White box | Baseline interprétable |

### Pourquoi ces trois modèles ?

- **Random Forest et XGBoost** sont retenus pour leur performance supérieure (F1 ≈ 0.80) et parce que leur opacité rend le XAI précisément nécessaire : sans outil d'explication, leurs décisions sont ininterprétables.
- **Decision Tree** sert de *baseline white box* : si LIME, SHAP et MC-LIME désignent les mêmes features importantes que les nœuds de l'arbre, cela **valide la cohérence des explications post-hoc**.

### Modèles exclus

- **Gradient Boosting** : quasi-identique à XGBoost (F1 = 0.7987, même famille algorithmique) — redondant.
- **MLP** : F1 de 0.503 sur 128 étudiants — sous-apprentissage avéré, explication non pertinente.
- **KNN, Naive Bayes** : performances insuffisantes.

---

## 4. Sélection des instances (filtre de consensus)

### Principe

Pour garantir que les explications XAI portent sur des prédictions **fiables et correctes**, un filtre à deux niveaux est appliqué.

### Niveau 1 — Consensus RF + XGBoost

Une instance n'est éligible à l'analyse XAI que si **Random Forest ET XGBoost la prédisent correctement simultanément**.

> Decision Tree n'est **pas** inclus dans ce filtre (F1 = 0.736 → taux d'erreur ~27%), mais reste utilisé pour l'analyse XAI une fois les instances sélectionnées.

**Résultat avant/après ajustement du filtre :**

| Profil | Avant ajustement | Après ajustement | Objectif |
|--------|-----------------|-----------------|---------|
| E+P+ | 1 instance | 4 instances | 4 |
| E+P− | 0 instance | 2 instances | 4 |
| E−P+ | 6 instances | 4 instances | 4 ✓ |
| E−P− | 1 instance | 3 instances | 4 |
| **TOTAL** | **8** | **13** | **16** |

### Niveau 2 — Filtre sur les prédictions correctes

Dans la méthode `generate_explanations()` de chaque module XAI, un filtre supplémentaire ignore toute explication générée sur une instance **mal prédite par le modèle concerné** (`vrai_label ≠ predit_label`).

Expliquer une prédiction erronée serait pédagogiquement contre-productif : le modèle a tort, donc son explication n'a pas de sens interprétatif.

**Impact sur Decision Tree :**

| Modèle | Instances sélectionnées | Après filtre | Ignorées |
|--------|------------------------|-------------|---------|
| Random Forest | 13 | 13 | 0 |
| XGBoost | 13 | 13 | 0 |
| Decision Tree | 13 | 11 | 2 |

---

## 5. Méthodes XAI implémentées

### 5.1 LIME — Local Interpretable Model-agnostic Explanations

LIME génère des perturbations locales autour d'une instance cible et approche le modèle par un **modèle linéaire local**.

**Paramètres clés :**

| Paramètre | Valeur | Justification |
|-----------|--------|--------------|
| `num_samples` | 10 000 | Augmenté depuis 5 000 pour stabiliser les explications (+116% de stabilité sur RF) |
| `num_features` | 10 | Features affichées par explication |

**Métriques après ajustement :**

| Modèle | Stabilité avant | Stabilité après | Gain |
|--------|----------------|----------------|------|
| Decision Tree | 0.316 | 0.365 | +15% |
| Random Forest | 0.351 | 0.759 | +116% |
| XGBoost | 0.330 | 0.700 | +112% |

> ℹ️ **Note sur la fidélité locale** : La fidélité reste faible (0.000–0.056) même après ajustement. C'est un résultat structurel : LIME approxime les frontières non-linéaires de RF/XGBoost par un modèle linéaire, ce qui produit nécessairement une approximation imparfaite. Ce résultat est documenté dans *Evaluating the Explainers* (Swamy et al., 2022) et motive l'évaluation par LLM-as-a-Judge en Phase 3.

---

### 5.2 MC-LIME — Monte Carlo LIME

Variante de LIME intégrant une composante de robustesse par tirages Monte Carlo.

**Paramètres clés :**

| Paramètre | Valeur |
|-----------|--------|
| `num_samples` | 10 000 |
| `num_features` | 15 (augmenté depuis 10) |

---

### 5.3 SHAP — SHapley Additive exPlanations

SHAP attribue à chaque feature une contribution marginale à la prédiction, fondée sur la théorie des jeux coopératifs (valeurs de Shapley).

---

### 5.4 DiCE — Diverse Counterfactual Explanations

DiCE génère des **explications contrefactuelles** : « Qu'est-ce qui devrait changer dans le profil de cet étudiant pour que la prédiction soit différente ? »

**Méthode retenue : `genetic`**

| Méthode | Principe | Résultat |
|---------|---------|---------|
| `kdtree` | Cherche parmi les vraies instances | ✗ Instable avec données normalisées |
| `random` | Génération aléatoire | ✗ Contrefactuels peu réalistes |
| `genetic` | Algorithme génétique | ✓ Stable + contrefactuels diversifiés |

**Paramètres ajoutés :**
- `proximity_weight=0.5` — pénalise les contrefactuels trop éloignés de l'instance originale (réalisme)
- `diversity_weight=1.0` — favorise des contrefactuels variés (richesse informationnelle)

---

### 5.5 CEM — Abandonné

CEM (Contrastive Explanations Method) n'a pas pu être intégré en raison d'incompatibilités de dépendances entre `alibi`, `TensorFlow` et l'environnement Python utilisé (voir [section 6.5](#65-cem--abandon)).

Les quatre méthodes retenues — LIME, MC-LIME, SHAP, DiCE — couvrent les trois catégories d'explications XAI : importance de features (LIME, MC-LIME, SHAP) et contrefactuelle (DiCE).

---

## 6. Ajustements techniques documentés

### 6.1 SMOTE — `k_neighbors=3`

Le paramètre par défaut de SMOTE est `k_neighbors=5`. Or le profil E+P− ne compte que 15 étudiants dans le train (split 80/20). Avec k=5, SMOTE cherche 5 voisins proches pour générer des exemples synthétiques — impossible si la classe n'a que 15 instances.

**Ajustement :** `k_neighbors` réduit à **3**. Ce paramètre est validé dans la littérature pour les petits datasets déséquilibrés.

---

### 6.2 Suppression de `TOTAL_STUDENTS_NUMBER`

Cette variable apparaissait comme feature importante dans les explications LIME et MC-LIME. Or elle est **constante** sur toute la promotion 2021–2022 (128 étudiants = même cours).

**Risque identifié :** Si différentes sessions ont des effectifs différents, le modèle apprend à utiliser cet effectif administratif pour prédire le profil — il s'agit d'une **fuite de données (data leakage)**.

**Correction :** Ajout de `'TOTAL_STUDENTS'` dans la liste `mots_cles_a_supprimer` du preprocessing. Toutes les variantes (`TOTAL_STUDENTS_NUMBER`, `TOTAL_STUDENTS_NUMBER.1`, etc.) sont supprimées. Le preprocessing a été relancé et `evaluator.pkl` régénéré.

---

### 6.3 DiCE — Passage de `kdtree` à `genetic`

La méthode `kdtree` de DiCE est instable avec des données normalisées par `StandardScaler` : les valeurs normalisées créent des ambiguïtés dans la recherche par arbre KD.

**Solution :** Méthode `genetic` adoptée — stable sur données normalisées, contrefactuels diversifiés.

---

### 6.4 LIME/MC-LIME — Augmentation de `num_samples`

Avec la valeur par défaut (`num_samples=5000`), les métriques de stabilité étaient inacceptables (RF : 0.351, XGBoost : 0.330).

**Ajustement :** `num_samples` porté à **10 000**. Le doublement des perturbations locales améliore drastiquement la reproductibilité des explications (+116% sur RF) sans impact significatif sur le temps de calcul pour un dataset de 128 étudiants.

---

### 6.5 CEM — Abandon

| Tentative | Résultat |
|-----------|---------|
| `pip install alibi` (Python 3.9) | ✗ `thinc` incompatible |
| `pip install alibi==0.9.4 --no-deps` (Python 3.9) | ✗ module `spacy` manquant |
| Environnement conda Python 3.10 (machine 2) | `alibi` installé ✓ mais TF incompatible |
| `tensorflow-macos` sur Mac ARM (Python 3.11) | ✗ Kernel crash — conflits numpy/keras |
| **Décision finale** | CEM abandonné — justifié dans le mémoire |

---

## 7. Architecture du code

### Classe mère `XAIEvaluator`

Pour éviter la duplication de logique entre les modules LIME, MC-LIME, SHAP et DiCE, une **classe mère `XAIEvaluator`** a été créée. Toutes les classes XAI en héritent.

**Ce qu'elle centralise :**

- Constantes partagées : `SELECTED_MODELS`, `CONSENSUS_MODELS`, `CLASS_NAMES`
- Attributs communs : `X_train`, `X_test`, `y_train`, `y_test`, `feature_names`, `class_mapping`, `reverse_mapping`, `classes`
- Méthode `select_samples()` — une seule implémentation pour les 4 méthodes

**Principe DRY appliqué :** tout changement (modifier `CONSENSUS_MODELS`, renommer une classe) se propage automatiquement aux quatre modules sans édition manuelle répétée.

### Structure des notebooks

```
project/
├── preprocessing/
│   └── preprocessing.ipynb       # Nettoyage, SMOTE, normalisation
├── models/
│   └── training.ipynb            # Entraînement des 11 modèles
├── xai/
│   ├── XAIEvaluator.py           # Classe mère partagée
│   ├── lime_explainer.ipynb
│   ├── mclime_explainer.ipynb
│   ├── shap_explainer.ipynb
│   └── dice_explainer.ipynb
└── evaluation/
    └── llm_judge.ipynb            # Phase 3 — LLM-as-a-Judge
```

---

## 8. Limitations connues

| Limitation | Nature | Impact |
|-----------|--------|--------|
| Dataset de 128 étudiants | Structurelle | Sous-représentation de certains profils (E+P− : 2 instances XAI) |
| Fidélité LIME faible (0.000–0.056) | Structurelle | LIME approxime imparfaitement les frontières non-linéaires de RF/XGBoost |
| E+P− : 2 instances seulement | Structurelle | Résultats non généralisables pour ce profil |
| CEM non implémenté | Technique | Catégorie contrastive partiellement couverte par DiCE |
| Données d'une seule promotion (2021–2022) | Structurelle | Généralisation limitée à d'autres contextes pédagogiques |

---

## 9. Glossaire

| Terme | Définition |
|-------|-----------|
| **XAI** | Explainable AI — ensemble de méthodes visant à rendre les décisions des modèles ML interprétables |
| **LIME** | Local Interpretable Model-agnostic Explanations — explication locale par approximation linéaire |
| **SHAP** | SHapley Additive exPlanations — attribution des contributions de chaque feature par théorie des jeux |
| **DiCE** | Diverse Counterfactual Explanations — explications du type « que faudrait-il changer ? » |
| **MC-LIME** | Variante de LIME avec robustesse Monte Carlo |
| **CEM** | Contrastive Explanations Method — non intégré (incompatibilités dépendances) |
| **Black box** | Modèle dont le processus de décision interne est opaque (ex: Random Forest, XGBoost) |
| **White box** | Modèle dont la logique de décision est directement lisible (ex: Decision Tree) |
| **Consensus** | Critère exigeant que RF et XGBoost prédisent correctement une instance pour qu'elle soit éligible |
| **Data leakage** | Fuite d'information du futur ou de l'environnement dans le modèle, biaisant ses prédictions |
| **SMOTE** | Synthetic Minority Over-sampling Technique — génération d'exemples synthétiques pour rééquilibrer les classes |
| **F1-Score** | Moyenne harmonique précision/rappel — métrique privilégiée sur données déséquilibrées |
| **AUC-ROC** | Aire sous la courbe ROC — mesure la capacité discriminante du modèle |
| **LLM-as-a-Judge** | Protocole d'évaluation utilisant un LLM pour noter la qualité des explications XAI (Phase 3) |

---

*Documentation générée dans le cadre du mémoire de Master — XAI en contexte éducatif, 2024–2025.*
