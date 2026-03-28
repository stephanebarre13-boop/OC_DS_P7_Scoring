# 💳 Modèle de scoring crédit — Prédiction du risque client

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikitlearn)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🎯 Objectif

Développer un **modèle de scoring crédit orienté décision métier** pour prédire le risque de défaut de paiement d'un client, en optimisant le seuil de classification selon le coût business réel — et non selon une métrique technique standard.

---

## 📊 Résultats

| Indicateur | Valeur |
|-----------|--------|
| Modèle retenu | LightGBM |
| AUC-ROC | 0.78 |
| Seuil optimisé (coût métier) | Personnalisé |
| Gestion déséquilibre de classes | SMOTE + class_weight |
| Tracking expériences | MLflow |
| Dashboard interactif | Streamlit |

---

## 🏗️ Architecture du projet

```
Données clients (Home Credit)
        │
        ▼
  Exploration & nettoyage
  Feature engineering
        │
        ▼
  Modélisation ML
  (LightGBM, Random Forest,
   Régression Logistique)
        │
        ▼
  Optimisation seuil
  (coût métier : FN > FP)
        │
        ▼
  Tracking MLflow
        │
        ▼
  Dashboard Streamlit
  (scoring client + explication)
```

---

## ⚙️ Stack technique

| Composant | Technologie |
|-----------|------------|
| Modélisation | LightGBM · Random Forest · Régression Logistique |
| Déséquilibre de classes | SMOTE · class_weight |
| Optimisation seuil | Fonction de coût métier personnalisée |
| Tracking | MLflow |
| Dashboard | Streamlit |
| Interprétabilité | SHAP |
| Données | Home Credit Default Risk (Kaggle) |

---

## 🔑 Choix techniques clés

**Optimisation du seuil de décision**
Le seuil de classification n'est pas fixé à 0.5 par défaut — il est optimisé selon une **fonction de coût métier** où un faux négatif (accorder un crédit à un mauvais payeur) coûte plus cher qu'un faux positif (refuser un bon client). Cette approche est directement applicable en contexte bancaire réel.

**Gestion du déséquilibre de classes**
Le dataset Home Credit est fortement déséquilibré (~8% de défauts). Combinaison de SMOTE et class_weight pour corriger ce biais sans perdre d'information.

**Interprétabilité SHAP**
Les décisions du modèle sont expliquées via SHAP — indispensable en contexte réglementaire bancaire (obligation d'expliquer les refus de crédit).

---

## 🖥️ Dashboard Streamlit

Le dashboard permet de :
- Saisir les caractéristiques d'un client
- Obtenir un **score de risque** en temps réel
- Visualiser les **variables les plus influentes** (SHAP)
- Ajuster le **seuil de décision** selon le profil de risque souhaité

---

## 🗂️ Structure du projet

```
OC_DS_P7_Scoring/
│
├── notebooks/
│   └── P7_Scoring_EDA.ipynb           # Exploration des données
│   └── P7_Scoring_Modelisation.ipynb  # Modélisation & optimisation
│
├── src/
│   └── dashboard.py                   # Application Streamlit
│   └── scoring_model.py               # Pipeline ML
│
├── mlruns/                            # Expériences MLflow
│
└── README.md
```

---

## 📚 Formation

Projet réalisé dans le cadre de la formation **Data Scientist** — [OpenClassrooms](https://openclassrooms.com)
Accréditation universitaire **WSCUC** (Western Association of Schools and Colleges — USA) · Niveau Master / Bac+5

---

## 👤 Auteur

**Stéphane Barré**
Data Scientist | PySpark · AWS · ML · NLP | Double profil Ingénieur · Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-stephane--barre--data-blue?logo=linkedin)](https://www.linkedin.com/in/stephane-barre-data)
[![GitHub](https://img.shields.io/badge/GitHub-stephanebarre13--boop-black?logo=github)](https://github.com/stephanebarre13-boop)
