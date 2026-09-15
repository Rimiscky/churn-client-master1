# Prédiction du Churn Client

Projet #3 - Introduction au Machine Learning - L'École Multimédia.

## Objectif

L'objectif est de prédire si un client d'une entreprise de télécommunications risque de résilier son abonnement (`Churn = Yes`).

Le projet reste volontairement simple et pédagogique : trois modèles classiques sont comparés afin de comprendre leurs différences et de pouvoir expliquer les résultats facilement à l'oral.

- Régression logistique
- Arbre de décision
- Random Forest

## Données

Dataset : **Telco Customer Churn** de Kaggle.

Le fichier contient 7 043 clients et 21 colonnes. Le taux de churn est d'environ 26,5 %.

Le CSV brut n'est pas versionné dans GitHub. Il doit être placé dans :

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Structure

```text
.
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_exploration_preparation.ipynb
│   └── 02_modelisation_evaluation.ipynb
├── src/
│   ├── train.py
│   └── visualize.py
├── RESULTATS.md
├── RAPPORT_FINAL.md
├── GUIDE_ORAL.md
├── REVUE_CODE.md
├── tests/
│   └── test_train.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Exécution

```bash
python src/train.py
```

Le script lance la recherche d'hyperparamètres sur cinq plis, puis évalue les modèles optimisés. Le notebook 2 réutilise ces mêmes fonctions et lit directement le CSV brut. Après une modification de `src/train.py`, redémarrer le noyau du notebook et réexécuter les cellules pour recharger le code.

## Tests de cohérence

```bash
python -m unittest discover -s tests -v
```

Les tests utilisent des données fictives : conservation des clients distincts, validation de la cible, valeurs manquantes et catégories nouvelles. Ils comparent aussi les exports du script et des cellules du notebook avec des grilles réduites. Ils ne valident pas les scores sur Telco. Le workflow GitHub Actions est configuré pour exécuter ces tests.

## Préparation des données

Les étapes principales sont :

1. supprimer les doublons exacts en conservant `customerID` lors de la comparaison ;
2. convertir `TotalCharges` en nombre ;
3. retirer `customerID` des variables explicatives et convertir la cible ;
4. séparer les données en 80 % entraînement / 20 % test, avec stratification ;
5. apprendre l'imputation, l'encodage et la standardisation dans le pipeline, sur les seules données d'entraînement de chaque pli.

Deux clients distincts avec le même profil sont conservés. Sans colonne `customerID`, aucune déduplication automatique n'est appliquée.

## Pourquoi plusieurs métriques ?

Le dataset est déséquilibré : environ trois clients sur quatre ne churnent pas.

L'Accuracy seule n'est donc pas suffisante. On regarde aussi :

- **Precision** : parmi les clients prédits comme churners, combien le sont vraiment ?
- **Recall** : parmi les vrais churners, combien sont détectés ?
- **F1-score** : compromis entre Precision et Recall ;
- **ROC-AUC** : capacité globale du modèle à séparer churners et non-churners.

## Résultats reproductibles

Le script et le notebook 2 utilisent les mêmes fonctions pour nettoyer les données, construire les modèles, rechercher les hyperparamètres et exporter les résultats.

Après exécution :

- `results/metrics.csv` contient les métriques sur le jeu de test ;
- `results/best_parameters.json` contient les paramètres sélectionnés et le ROC-AUC de validation croisée.

Les résultats ont été recalculés sur le CSV retrouvé le 15 septembre 2026. `RESULTATS.md` contient les chiffres actuels ; `RESULTATS_ARCHIVE.md` conserve les anciens chiffres. La régression logistique est privilégiée pour sa simplicité, avec un ROC-AUC de 0,841 contre 0,842 pour la Random Forest.

## Hyperparamètres testés

Les grilles sont définies dans `get_searches()` de `src/train.py`, utilisé aussi par le notebook :

- Régression logistique : `C = 0.1, 1, 10`.
- Arbre : `max_depth = 3, 5, 7` et `min_samples_leaf = 10, 20, 40`.
- Random Forest : `n_estimators = 200, 300`, `max_depth = 8, None` et `min_samples_leaf = 1, 3`.

Les meilleurs réglages sont déterminés par validation croisée à cinq plis sur l'entraînement, avec le ROC-AUC comme critère. Ils sont exportés dans `results/best_parameters.json`.

## Visuels pour la soutenance

```bash
python -m src.visualize
```

Cette commande réentraîne les modèles avec les grilles communes, actualise les résultats et exporte quatre figures en PNG et PDF dans `results/figures/` : répartition du churn, comparaison avec la référence naïve, matrice de confusion et coefficients de la régression logistique.

La dernière section du notebook 2 utilise les modèles déjà entraînés pour produire les mêmes graphiques. Le guide `VISUELS_SOUTENANCE.md` explique quoi dire à l'oral et comment interpréter les coefficients sans conclure à une causalité.

Le manifeste `results/visualisation_manifest.json` est créé par la commande ci-dessus et identifie son exécution (CSV, versions et paramètres). Les cellules du notebook exportent les figures ; elles ne réécrivent pas ce manifeste.

## Déploiement et suivi envisagés

`DEPLOIEMENT_MONITORING.md` décrit une utilisation sur de nouveaux fichiers clients, les éléments à sauvegarder, les contrôles d'entrée, le suivi et les conditions de réentraînement. Il contient un schéma et un texte pour la soutenance.

Il s'agit d'une proposition : le projet exécute actuellement l'entraînement et l'évaluation en local ; la sauvegarde du pipeline, le parcours de prédiction sur nouveaux clients et la surveillance restent à implémenter.

## Conclusion simple

Le projet montre qu'il est possible d'identifier une partie importante des clients susceptibles de résilier leur abonnement.

La régression logistique est privilégiée pour sa simplicité. Ses performances recalculées sont proches de celles de la Random Forest sur ce découpage.

Dans un contexte réel, l'entreprise pourrait utiliser ce score de risque pour contacter en priorité les clients les plus susceptibles de partir.

## Documents du rendu

- `notebooks/01_exploration_preparation.ipynb` : qualité des données, statistiques, visualisations et nettoyage ;
- `notebooks/02_modelisation_evaluation.ipynb` : modèles, comparaison, optimisation simple et validation ;
- `RAPPORT_FINAL.md` : synthèse du projet ;
- `GUIDE_ORAL.md` : aide pour présenter le projet simplement ;
- `REVUE_CODE.md` : revue de code en deux minutes, outils, tests et questions du professeur ;
- `RESULTATS.md` : résultats chiffrés recalculés ;
- `VISUELS_SOUTENANCE.md` : les quatre figures et leurs explications pour l’oral ;
- `DEPLOIEMENT_MONITORING.md` : proposition de déploiement et de suivi, avec texte oral.

## Conventions Git

Les commits suivent Conventional Commits : `feat:`, `fix:`, `docs:`, `chore:`.
