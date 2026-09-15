# Guide oral - Projet Churn

Ce document sert à expliquer le projet simplement, sans apprendre des phrases trop techniques.

## 1. Présenter le problème

> Mon objectif est de prédire quels clients risquent de résilier leur abonnement. La variable cible s'appelle `Churn` et elle contient deux valeurs : Yes ou No.

À retenir : c'est une **classification binaire** parce qu'il y a deux classes possibles.

## 2. Présenter les données

> Le dataset contient 7 043 clients et 21 colonnes. Il y a des informations sur les contrats, les services utilisés, l'ancienneté et les montants facturés.

> Environ 26,5 % des clients ont churné.

À retenir : les classes ne sont pas équilibrées, donc l'Accuracy seule ne suffit pas.

## 3. Expliquer le nettoyage

> J'ai supprimé `customerID` parce que c'est seulement un identifiant. J'ai aussi converti `TotalCharges` en numérique. Quelques cellules étaient vides, donc elles sont devenues des valeurs manquantes. Je les ai remplacées par la médiane.

> Les variables texte ont été transformées en colonnes numériques avec le One-Hot Encoding.

## 4. Expliquer le train/test

> J'ai séparé les données en deux parties : 80 % pour entraîner le modèle et 20 % pour le tester.

> Le jeu de test n'est pas utilisé pendant l'entraînement. Il sert à vérifier si le modèle fonctionne sur des données qu'il n'a jamais vues.

## 5. Montrer que j'ai compris avec une baseline

> Avant les vrais modèles, j'ai créé une baseline naïve qui prédit toujours la classe majoritaire, donc "pas de churn".

> Comme environ 73,5 % des clients ne churnent pas, cette baseline peut avoir une Accuracy correcte. Mais son Recall est nul, donc elle ne détecte aucun churner.

À retenir : **une Accuracy correcte ne veut pas forcément dire qu'un modèle est utile**.

## 6. Expliquer les trois modèles

### Régression logistique

> C'est mon modèle principal. Il estime la probabilité qu'un client churn. Il est simple et facile à interpréter.

### Arbre de décision

> Il fonctionne comme une suite de questions : par exemple, quel type de contrat ? quelle ancienneté ? puis il arrive à une décision.

### Random Forest

> C'est un ensemble de plusieurs arbres de décision. Chaque arbre donne son avis et la forêt combine leurs décisions.

## 7. Expliquer les métriques

### Accuracy

> C'est la proportion totale de bonnes prédictions.

### Precision

> Parmi les clients que le modèle annonce comme churners, combien churnent réellement ?

### Recall

> Parmi tous les clients qui ont vraiment churné, combien le modèle a réussi à détecter ?

Pour ce projet, le Recall est important parce qu'on veut éviter de rater trop de clients à risque.

### F1-score

> C'est un compromis entre Precision et Recall.

### ROC-AUC

> C'est une mesure globale de la capacité du modèle à différencier les churners des non-churners. Plus le score est proche de 1, mieux c'est.

## 8. Expliquer la matrice de confusion

- vrai positif : le modèle prédit churn et le client churn vraiment ;
- faux positif : le modèle prédit churn mais le client reste ;
- faux négatif : le modèle prédit qu'il reste mais le client churn ;
- vrai négatif : le modèle prédit qu'il reste et il reste vraiment.

> Pour l'entreprise, le faux négatif peut être plus coûteux car elle ne détecte pas un client qui va partir.

## 9. Montrer que j'ai compris le sur-apprentissage

> J'ai comparé les performances sur le train et sur le test.

> Si un modèle est excellent sur le train mais beaucoup moins bon sur le test, cela peut vouloir dire qu'il a trop appris les données d'entraînement. C'est le sur-apprentissage.

À retenir : un bon modèle doit fonctionner aussi sur des données qu'il n'a jamais vues.

## 10. Expliquer l'optimisation

> J'ai utilisé GridSearchCV pour tester quelques valeurs d'hyperparamètres automatiquement.

> Je n'ai pas testé énormément de combinaisons, car le but du projet est surtout de comprendre le processus.

### Validation croisée

> Au lieu de tester un réglage sur un seul découpage, la validation croisée fait plusieurs découpages du jeu d'entraînement et calcule une moyenne.

## 11. Expliquer le seuil 40 %, 50 %, 60 %

> La régression logistique donne une probabilité de churn. Par défaut, on utilise un seuil de 50 %.

> J'ai aussi testé 40 % et 60 % pour voir l'effet sur Precision et Recall.

> Si je baisse le seuil, je détecte plus de churners mais je crée plus de fausses alertes. Si je monte le seuil, je signale moins de clients mais je risque d'en manquer davantage.

À retenir : **le seuil dépend du besoin métier**.

## 12. Résultats recalculés à retenir

- Régression logistique : accuracy 74,0 %, rappel 78,3 %, précision 50,6 %, ROC-AUC 0,841.
- Random Forest : rappel 79,4 %, ROC-AUC 0,842.
- Référence naïve : accuracy 73,5 %, mais aucun départ détecté.

> Ma régression logistique détecte 293 départs sur 374. Elle manque 81 départs et produit 286 fausses alertes. Le rappel seul ne suffit donc pas à juger son utilité.

## 13. Pourquoi privilégier la régression logistique ?

> Son score est proche de celui de la Random Forest sur ce test. Je la privilégie pour sa simplicité. Le petit écart de ROC-AUC ne prouve pas qu'un modèle est significativement meilleur.

## 14. Lire les coefficients

Montrer `04_coefficients_logistique.png`. Un coefficient positif contribue à augmenter le score ; un coefficient négatif contribue à le réduire dans ce modèle.

> Les coefficients sont des associations, pas des causes. Certaines variables se recoupent, par exemple les frais et les services. Un coefficient négatif des frais mensuels ne signifie pas qu'augmenter les prix fidélise les clients.

Toutes les modalités sont encodées, donc ne pas expliquer le coefficient d'une catégorie comme un effet face à une référence omise. Les variables numériques sont standardisées. Le guide `VISUELS_SOUTENANCE.md` développe ces limites.

## 15. Recommandation métier

> Le modèle pourrait servir à classer les clients par niveau de risque. L'équipe rétention pourrait ensuite contacter en priorité les clients les plus à risque avec une offre ou une enquête de satisfaction.

> Le modèle aide à prioriser, mais il ne remplace pas la décision humaine.

## 16. Limites du projet

> Le modèle apprend à partir de données historiques. Les comportements peuvent changer dans le temps.

> Le dataset ne contient pas toutes les raisons personnelles qui peuvent expliquer un départ.

> Le modèle détecte des associations, pas des causes.

> En production, il faudrait contrôler régulièrement les performances et réentraîner le modèle avec des données récentes.

## 17. Ce que j'ai appris

> J'ai compris qu'un projet Machine Learning ne consiste pas seulement à entraîner un modèle. Il faut aussi nettoyer les données, choisir les bonnes métriques, comparer train et test, analyser les erreurs et relier les résultats au besoin métier.

> J'ai aussi compris qu'un modèle plus complexe n'est pas forcément meilleur si un modèle simple donne presque les mêmes résultats.

## 18. Conclusion orale courte

> J'ai commencé par analyser et nettoyer les données. J'ai créé une baseline pour vérifier qu'une bonne Accuracy pouvait être trompeuse. J'ai ensuite comparé trois modèles, étudié leurs erreurs, vérifié le sur-apprentissage, testé quelques hyperparamètres et plusieurs seuils de décision. Je privilégie la régression logistique pour sa simplicité, avec des performances proches de la Random Forest sur ce test. Le modèle pourrait aider une entreprise télécom à identifier plus tôt les clients à risque.

## 19. Questions probables du jury

**Pourquoi supprimer customerID ?**

Parce que c'est un identifiant unique, pas une caractéristique du comportement du client.

**Pourquoi utiliser la médiane ?**

Parce qu'elle est moins sensible aux valeurs extrêmes que la moyenne.

**Pourquoi pas seulement l'Accuracy ?**

Parce que seulement environ 26,5 % des clients churnent. Une baseline qui prédit toujours "pas de churn" peut déjà avoir une Accuracy correcte tout en étant inutile.

**Pourquoi avoir créé une baseline ?**

Pour avoir un point de comparaison très simple et vérifier que mes vrais modèles apportent réellement quelque chose.

**Pourquoi regarder train et test ?**

Pour vérifier que le modèle généralise et qu'il ne mémorise pas seulement les données d'entraînement.

**Pourquoi tester plusieurs seuils ?**

Parce que le seuil change le compromis entre Precision et Recall. Le meilleur seuil dépend du coût métier des erreurs.

**Pourquoi la régression logistique ?**

Parce qu'elle est simple et interprétable. Son ROC-AUC est de 0,841 contre 0,842 pour la Random Forest sur ce test.

**Qu'est-ce qu'un hyperparamètre ?**

C'est un réglage choisi avant l'entraînement, par exemple la profondeur maximale d'un arbre.

**Qu'est-ce que le sur-apprentissage ?**

C'est quand un modèle apprend trop précisément les données d'entraînement et devient moins bon sur de nouvelles données.

**Est-ce que les variables importantes causent le churn ?**

Non. Elles sont associées au churn dans ce dataset, mais cela ne prouve pas une relation de cause à effet.


## 20. Revue de code à préparer (2 minutes)

Le document `REVUE_CODE.md` contient le parcours chronométré, un texte oral, le rôle des bibliothèques et les réponses aux questions de génie logiciel.

Ouvrir à l'avance `src/train.py` et `tests/test_train.py`. Montrer dans cet ordre : `main`, le nettoyage, le pipeline, l'évaluation et un test de conservation des clients. Les commentaires en français servent de repères pour expliquer les choix.

Retenir la différence entre tests logiciels et jeu de test ML : les premiers vérifient le programme, le second mesure la qualité des prédictions. Ne pas annoncer de réussite sur GitHub Actions sans avoir vérifié l'exécution distante.


## 21. Les quatre visuels

Utiliser dans cet ordre : répartition du churn, comparaison des modèles, matrice de confusion, coefficients. Le document `VISUELS_SOUTENANCE.md` contient les images et une phrase à dire pour chacune. Les chiffres proviennent de l'exécution du 15 septembre 2026.


## 22. Déploiement et monitoring (environ 90 secondes)

S'appuyer sur `DEPLOIEMENT_MONITORING.md` : il contient le schéma, un texte oral et les réponses aux questions du professeur.

À retenir : sauvegarder le pipeline entier ; recevoir un fichier sans cible ; produire un classement ; surveiller immédiatement les données et, plus tard, les performances quand les départs sont connus. Définir la période à prédire avant d'affirmer une capacité d'anticipation.

Présenter clairement cette partie comme une proposition. Aucun service de prédiction ni monitor actif n'a été déployé. Dans les deux dernières minutes de la présentation, consacrer environ 90 secondes à cette proposition et 30 secondes aux limites et à la conclusion.
