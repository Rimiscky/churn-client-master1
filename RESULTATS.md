# Résultats recalculés - Churn client

## Exécution du 15 septembre 2026

Les modèles ont été réentraînés avec les grilles communes au script et au notebook. Les figures sont issues de cette exécution réelle, sans données fictives.

Le CSV a été retrouvé dans le dossier Téléchargements puis copié dans `data/raw/`. Il contient 7 043 clients, 21 colonnes et 7 043 identifiants distincts. Après nettoyage : 7 043 lignes, dont 1 869 départs (26,54 %). Onze valeurs de `TotalCharges` deviennent manquantes et sont imputées dans le pipeline.

## Protocole

- Entraînement : 5 634 clients ; test : 1 409 clients, dont 374 départs.
- Séparation stratifiée 80/20, `random_state = 42`.
- Recherche sur cinq plis de l'entraînement, critère ROC-AUC.
- Classes pondérées avec `class_weight="balanced"`.
- Prédictions des classes au seuil par défaut du modèle.

## Scores sur le jeu de test

| Modèle | Accuracy | Précision | Rappel | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0,754 | 0,524 | 0,794 | 0,631 | 0,842 |
| Régression logistique | 0,740 | 0,506 | 0,783 | 0,615 | 0,841 |
| Arbre de décision | 0,752 | 0,521 | 0,781 | 0,625 | 0,832 |

La référence naïve obtient une accuracy de 0,735, un rappel nul et un ROC-AUC de 0,500. L'accuracy seule masque donc son incapacité à détecter les départs.

La Random Forest obtient ici un ROC-AUC de 0,842 contre 0,841 pour la régression logistique. Cet écart descriptif ne prouve pas un avantage statistiquement significatif. La régression logistique reste le modèle privilégié pour sa simplicité.

## Paramètres sélectionnés

### Régression logistique

- `C = 10`
- ROC-AUC moyen de validation : 0.845498.

### Arbre de décision

- `max_depth = 5`
- `min_samples_leaf = 40`
- ROC-AUC moyen de validation : 0.827368.

### Random Forest

- `max_depth = 8`
- `min_samples_leaf = 3`
- `n_estimators = 300`
- ROC-AUC moyen de validation : 0.847129.

## Matrice de confusion de la régression logistique

| Situation réelle | Prédit : reste | Prédit : part |
|---|---:|---:|
| Reste | 749 | 286 |
| Part | 81 | 293 |

Le modèle détecte 293 des 374 départs (rappel de 78,3 %). Parmi ses 579 alertes, 293 correspondent à un départ (précision de 50,6 %). Les fausses alertes et les départs manqués illustrent le compromis métier.

## Interprétation des coefficients

Le graphique présente les 12 coefficients de plus grande valeur absolue. Les variables numériques sont standardisées. Les catégories sont toutes encodées : le coefficient d'une modalité n'est pas un effet par rapport à une catégorie omise.

L'ancienneté, le contrat de deux ans et la modalité DSL ont des coefficients négatifs dans ce modèle ; la fibre et le contrat mensuel ont des coefficients positifs. Les variables corrélées peuvent se partager l'information et rendre les coefficients difficiles à interpréter isolément.

Le coefficient négatif des frais mensuels ne signifie donc pas qu'augmenter les prix réduit le churn. C'est une association conditionnelle dans ce modèle, avec les services, les contrats et les autres montants présents simultanément. Aucune conclusion causale n'est justifiée.

## Fichiers de preuve

- `results/metrics.csv` : métriques des trois modèles optimisés.
- `results/best_parameters.json` : réglages réellement sélectionnés.
- `results/figures/` : quatre figures PNG et PDF, avec les tableaux sources.
- `results/visualisation_manifest.json` : empreinte du CSV, versions, date et protocole.
- `RESULTATS_ARCHIVE.md` : anciens résultats, conservés uniquement pour historique.

Les chiffres de cette page remplacent les anciennes valeurs de l'arbre et de la Random Forest. Les résultats de la régression logistique sont confirmés à l'arrondi affiché.

## Limites

Les scores dépendent du découpage et de l'environnement consignés. Aucun intervalle de confiance n'est calculé. Le test de différents seuils dans le notebook est pédagogique ; une validation séparée est nécessaire pour choisir un seuil opérationnel. Les scores ne prouvent pas l'efficacité d'une campagne de rétention réelle.
