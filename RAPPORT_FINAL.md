# Rapport final - Prédiction du churn client

## 1. Contexte

L'objectif du projet est de construire un modèle de Machine Learning capable d'identifier les clients d'une entreprise de télécommunications qui risquent de résilier leur abonnement.

Le dataset utilisé est **Telco Customer Churn** de Kaggle. Chaque ligne représente un client et contient des informations sur son contrat, ses services, son ancienneté et ses frais.

La variable à prédire est `Churn` :

- `Yes` : le client a quitté l'entreprise ;
- `No` : le client est resté.

Il s'agit donc d'un problème de **classification binaire**.

## 2. Analyse de la qualité des données

Le dataset contient :

- 7 043 clients ;
- 21 colonnes ;
- aucun doublon exact ;
- 11 valeurs problématiques dans `TotalCharges` après conversion en nombre.

`TotalCharges` était lu comme du texte à cause de quelques cellules vides. Ces valeurs sont transformées en valeurs manquantes puis remplacées par la médiane dans le pipeline.

La colonne `customerID` est supprimée car elle identifie chaque client mais n'apporte pas d'information utile pour prédire le churn.

## 3. Analyse exploratoire

Le churn représente environ **26,5 %** des clients.

Cela signifie que les classes sont déséquilibrées : il y a beaucoup plus de clients qui restent que de clients qui partent.

L'analyse montre notamment que le churn varie selon :

- l'ancienneté du client ;
- le type de contrat ;
- le type de service Internet ;
- les frais mensuels ;
- les frais totaux.

Les contrats mensuels apparaissent davantage associés au churn, tandis que les contrats de deux ans sont davantage associés à la fidélité.

## 4. Préparation des données

Le nettoyage supprime les doublons strictement identiques avant de retirer l'identifiant. Il conserve donc deux clients distincts ayant les mêmes caractéristiques. `TotalCharges` est ensuite convertie en nombre et la cible en 0/1.

Après séparation stratifiée en 80 % entraînement et 20 % test, le pipeline apprend l'imputation médiane, l'imputation des catégories par leur valeur la plus fréquente, le One-Hot Encoding et la standardisation sur l'entraînement uniquement. Pendant la validation croisée, ces opérations sont réapprises dans chaque pli.

## 5. Modèles testés

Trois modèles ont été comparés :

### Régression logistique

C'est le modèle de référence. Il est simple, rapide et facile à interpréter.

### Arbre de décision

Il prend ses décisions sous forme de règles successives. Son fonctionnement est facile à visualiser, mais il peut sur-apprendre s'il devient trop profond.

### Random Forest

Il combine plusieurs arbres de décision. Il est généralement plus robuste, mais il est moins simple à expliquer.

## 6. Optimisation

Les grilles sont définies dans `get_searches()` de `src/train.py`, utilisé aussi par le notebook :

- Régression logistique : `C = 0.1, 1, 10`.
- Arbre : `max_depth = 3, 5, 7` et `min_samples_leaf = 10, 20, 40`.
- Random Forest : `n_estimators = 200, 300`, `max_depth = 8, None` et `min_samples_leaf = 1, 3`.

Les meilleurs réglages sont déterminés par validation croisée à cinq plis sur l'entraînement, avec le ROC-AUC comme critère. Ils sont exportés dans `results/best_parameters.json`.

## 7. Résultats recalculés le 15 septembre 2026

| Modèle | Accuracy | Précision | Rappel | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0,754 | 0,524 | 0,794 | 0,631 | 0,842 |
| Régression logistique | 0,740 | 0,506 | 0,783 | 0,615 | 0,841 |
| Arbre de décision | 0,752 | 0,521 | 0,781 | 0,625 | 0,832 |

Ces valeurs sont issues de l'exécution sur le CSV Telco retrouvé. Les anciens chiffres sont archivés dans `RESULTATS_ARCHIVE.md`.

## 8. Modèle privilégié

La régression logistique est privilégiée pour sa simplicité. Son ROC-AUC de 0,841 est proche de celui de la Random Forest (0,842), sans démonstration de différence statistiquement significative. Elle détecte 293 départs sur 374, mais produit aussi 286 fausses alertes.

## 9. Validation et généralisation

La recherche des hyperparamètres utilise uniquement le jeu d'entraînement, avec cinq plis. Le jeu de test mesure les performances des modèles obtenus. Les paramètres exacts et scores de validation sont consignés dans `results/best_parameters.json`.

Le notebook montre aussi les effets de plusieurs seuils à titre pédagogique. Pour sélectionner un seuil, il faudrait utiliser une validation séparée. La proximité des scores moyens de validation et de test ne suffit pas à prouver l'absence de sur-apprentissage.

## 10. Interprétation et visuels

Quatre graphiques sont disponibles dans `results/figures/` : répartition du churn, comparaison des modèles, matrice de confusion et coefficients de la régression logistique. Ils reprennent les calculs réels.

Les coefficients montrent des associations conditionnelles. Ils dépendent de la standardisation et des variables corrélées ; les catégories sont toutes encodées. Le coefficient négatif des frais mensuels ne permet pas d'affirmer qu'une hausse des prix réduit les départs.

L'entreprise pourrait utiliser les scores pour prioriser les contacts, après définition du coût des erreurs et validation de cette utilisation sur des données récentes.

## 11. Limites

Le modèle ne dit pas pourquoi un client veut partir. Il détecte seulement des profils ressemblant aux clients qui ont churné dans les données historiques.

Les relations observées ne doivent pas être interprétées comme des relations de cause à effet.

## 12. Possibilité de déploiement

La solution envisagée est un traitement hebdomadaire de fichiers de clients actifs. Le pipeline complet serait sauvegardé puis appliqué sans réentraînement aux nouveaux fichiers, après contrôle du schéma. La sortie contiendrait l'identifiant, le score, la priorité, la date et la version du modèle.

Cette étape nécessite encore une sauvegarde du pipeline et un parcours de prédiction acceptant des clients sans colonne `Churn`. Le projet actuel ne comporte pas de service déployé. Avant une utilisation réelle, il faudrait définir l'instant de prédiction et la période cible, puis valider sur une période ultérieure : le test aléatoire Telco ne prouve pas l'anticipation des départs futurs.

## 13. Monitoring proposé

À chaque traitement, surveiller les échecs, le nombre de lignes, les valeurs manquantes, les catégories inconnues et la distribution des scores. Lorsque les départs réels sont connus pour la période cible, mesurer le rappel, la précision et les erreurs, en tenant compte des effectifs.

Une anomalie doit déclencher une analyse avant tout réentraînement. Comparer ensuite un éventuel nouveau modèle à l'ancien sur une période réservée, et conserver la possibilité de revenir à la version précédente. Aucun monitor ni traitement programmé n'est actif dans le projet.

Le document `DEPLOIEMENT_MONITORING.md` précise le parcours, les entrées et sorties, les responsabilités et les réponses pour l'oral.

## 14. Conclusion

Le projet montre qu'une approche simple de Machine Learning permet déjà d'identifier une part importante des clients à risque.

La régression logistique offre ici des performances proches de la Random Forest avec un modèle plus simple à présenter.

Le faible écart observé avec la Random Forest doit être évalué avec davantage de recul avant de conclure à un gain utile.
