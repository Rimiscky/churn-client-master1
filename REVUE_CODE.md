# Revue de code - Soutenance Churn

## Objectif

Présenter en deux minutes l'organisation du code, les choix de traitement et les preuves de vérification. Les détails suivants servent ensuite aux questions du professeur.

Le fichier principal est `src/train.py`. Les notebooks l'importent pour réutiliser ses fonctions. Le notebook 1 explore les données ; le notebook 2 ajoute les graphiques, une baseline et une comparaison pédagogique avant optimisation.

## 1. Parcours de démonstration en deux minutes

Ouvrir les fichiers avant la présentation. Utiliser la recherche du nom de fonction pour aller directement au passage utile.

| Temps | Passage à montrer | Explication à donner |
|---|---|---|
| 0:00 à 0:20 | `src/train.py`, fonction `main` | Cette fonction enchaîne les étapes. Les opérations sont séparées en fonctions pour être réutilisées et testées. |
| 0:20 à 0:45 | `clean_data` puis `split_features_target` | Je conserve les clients distincts lors du nettoyage, puis je retire leur identifiant des variables explicatives. La cible est séparée des caractéristiques. |
| 0:45 à 1:15 | `build_preprocessor` puis `optimize_models` | Le pipeline apprend l'imputation et l'encodage sur l'entraînement de chaque pli. Les réglages sont comparés par validation croisée. |
| 1:15 à 1:35 | `evaluate_model` et `save_results` | Les prédictions sont évaluées sur le test. Les scores et les réglages sont enregistrés pour éviter une recopie manuelle incohérente. |
| 1:35 à 2:00 | `tests/test_train.py`, test `test_distinct_clients_are_preserved` | Ce test reproduit un risque concret : supprimer deux clients différents qui ont le même profil. Les autres tests vérifient les erreurs de données et la cohérence script/notebook. |

Ne pas lancer une recherche complète d'hyperparamètres pendant ces deux minutes. Montrer la sortie de tests déjà obtenue ou lancer uniquement le test court ci-dessous.

## 2. Texte à répéter à l'oral

> Mon code est organisé autour d'un script commun et de deux notebooks. Le script contient les fonctions réutilisables ; les notebooks servent à explorer les données et à présenter les résultats.
>
> Dans la fonction main, on retrouve le déroulement du projet : charger les données, les nettoyer, séparer les caractéristiques de la cible, créer les jeux d'entraînement et de test, optimiser les modèles puis les évaluer.
>
> Un point important du nettoyage concerne les doublons. Deux clients peuvent avoir les mêmes caractéristiques. Je conserve donc l'identifiant pour comparer les lignes avant de le retirer de l'apprentissage.
>
> Le prétraitement est placé dans un pipeline. La médiane et les transformations sont apprises uniquement sur l'entraînement, y compris à l'intérieur de chaque pli de validation croisée. Cela évite d'utiliser les données de validation pour préparer le modèle.
>
> Le script et le notebook appellent la même fonction d'optimisation. Ils exportent les scores et les paramètres calculés, ce qui limite les différences entre les documents et le code.
>
> Enfin, les tests vérifient des comportements concrets : conserver les clients distincts, refuser une cible invalide et accepter une catégorie nouvelle. Un test compare aussi les exports du script et des cellules du notebook sur des données fictives. Ces tests vérifient le fonctionnement du programme ; ils ne prouvent pas ses performances sur les vrais clients.

## 3. Comprendre les fonctions

| Fonction | Entrée | Sortie ou effet | Pourquoi la séparer ? |
|---|---|---|---|
| `load_data` | Chemin du CSV | Tableau pandas, ou erreur explicite | Identifier immédiatement un fichier ou une cible absents |
| `clean_data` | Tableau brut | Copie nettoyée avec identifiant conservé | Tester le nettoyage sans entraîner de modèle |
| `split_features_target` | Tableau nettoyé avec `Churn` en Yes/No | `X` sans cible ni identifiant ; `y` en 0/1 | Empêcher la cible de servir de variable explicative |
| `build_preprocessor` | Colonnes de `X` | Transformations non entraînées | Définir le traitement propre à chaque type de variable |
| `get_models` | Aucun argument | Trois modèles de départ | Partager les réglages de la comparaison initiale du notebook |
| `get_searches` | Aucun argument | Modèles et valeurs d'hyperparamètres à tester | Définir les grilles à un seul endroit |
| `optimize_models` | `X_train`, `y_train` | Meilleurs pipelines entraînés et résultats de validation | Comparer les réglages sans recevoir le jeu de test |
| `evaluate_model` | Pipeline entraîné et jeu de test | Métriques ; matrice de confusion affichée | Évaluer sans réentraîner |
| `save_results` | Métriques et paramètres sélectionnés | `metrics.csv` et `best_parameters.json` | Conserver les résultats calculés ensemble |
| `main` | Aucun argument | Exécution complète du script | Rendre le déroulement facile à lire |

### Exemple concret : nettoyer n'est pas entraîner

Convertir la chaîne vide de `TotalCharges` en valeur manquante n'apprend aucune statistique. En revanche, calculer la médiane dépend des clients observés : cette opération doit rester dans le pipeline appris sur l'entraînement.

### Exemple concret : deux clients identiques en apparence

Avec trois lignes `A`, `B`, `A`, dont les caractéristiques sont identiques, le nettoyage conserve `A` et `B` et supprime la répétition exacte de `A`. Sans identifiant, il conserve les profils identiques par prudence. Il ne résout pas les cas où un même identifiant possède des informations contradictoires.

## 4. Rôle des outils utilisés

| Outil | Usage réel dans le projet | Explication simple |
|---|---|---|
| pandas | Lecture CSV, tableaux, colonnes, nettoyage et export | Manipuler les données des clients |
| NumPy | Reconnaissance des types numériques, opérations numériques et données fictives des tests | Travailler avec les nombres et les tableaux |
| scikit-learn | Prétraitement, modèles, validation croisée et métriques | Construire et évaluer la chaîne d'apprentissage |
| Matplotlib et Seaborn | Graphiques dans les notebooks | Rendre les distributions et erreurs visibles |
| Jupyter | Exécution des cellules et présentation des analyses | Relier code, explications et graphiques |
| `pathlib` et `json` | Chemins de fichiers et export des paramètres | Organiser les entrées et sorties du programme |
| `unittest` | Tests automatisés et remplacements temporaires pour les tests | Vérifier que le code garde le comportement attendu |
| GitHub Actions | Workflow prévu à chaque push et pull request | Automatiser la syntaxe et les tests après publication sur GitHub |

Plotly est déclaré dans les dépendances, mais n'est pas utilisé par le code examiné. Ne pas présenter un dashboard interactif comme une fonctionnalité déjà réalisée.

## 5. Montrer les tests

Depuis la racine du projet, après installation des dépendances :

```bash
python -m unittest discover -s tests -v
```

Pour une démonstration courte, montrer seulement le test de conservation des clients :

```bash
python -m unittest discover -s tests -k test_distinct_clients_are_preserved -v
```

| Test | Erreur qu'il permet de repérer |
|---|---|
| Fichier absent | Message de chargement incompréhensible ou erreur non prévue |
| Colonne cible absente | CSV accepté alors qu'il ne contient pas la variable à prédire |
| Clients distincts | Suppression abusive de clients ayant le même profil |
| Profils sans identifiants | Suppression de lignes sans preuve qu'elles représentent le même client |
| Cible invalide | Valeur inattendue transformée silencieusement en classe |
| Valeurs manquantes et catégorie inconnue | Prédiction qui échoue, ou médiane d'entraînement incorrecte |
| Cohérence script/notebook | Différence entre les exports des deux parcours sur un cas fictif |

Le dernier test réduit les grilles et le nombre d'arbres pour rester rapide. Il exécute les cellules réelles de séparation, optimisation et export, mais pas le notebook entier ni ses graphiques. Les fichiers temporaires du test ne remplacent pas les résultats du projet.

### Tests logiciels et jeu de test : deux notions différentes

- Un **test logiciel** vérifie une règle de fonctionnement, par exemple conserver deux identifiants distincts.
- Le **jeu de test ML** mesure la qualité des prédictions sur des clients non utilisés pour l'entraînement.
- Des tests logiciels réussis ne garantissent pas un bon rappel ou un bon ROC-AUC.

## 6. Git, documentation et vérification automatique

Le workflow `.github/workflows/python-check.yml` vérifie la syntaxe puis lance les tests sous Python 3.11. Il est configuré pour les push et les pull requests. Une configuration présente dans un fichier ne prouve pas qu'une exécution distante a réussi : pour l'affirmer, montrer une exécution verte dans GitHub Actions.

Le dossier local fourni ne contient pas de répertoire `.git` à sa racine. L'historique du dépôt du projet n'a donc pas été vérifié ici. Si le professeur demande Git, présenter l'historique réel sur le dépôt d'origine : un changement précis, sa différence de code et son message de commit. Ne pas inventer de branches, de commits ou de travail collaboratif.

Le README explique l'installation et l'exécution. Les commentaires français expliquent les décisions dans le code. `RESULTATS.md` présente désormais les résultats recalculés sur le CSV retrouvé. Les anciennes valeurs sont conservées dans `RESULTATS_ARCHIVE.md`.

## 7. Questions probables du professeur

**Pourquoi ne pas tout laisser dans le notebook ?**

Les fonctions partagées limitent la duplication. Une correction du nettoyage ou des grilles profite au script et aux notebooks. Après modification du script, il faut redémarrer le noyau pour recharger le code importé.

**Pourquoi `df.copy()` ?**

Pour que le nettoyage ne modifie pas le tableau brut utilisé ailleurs dans l'exploration.

**Quelle différence entre `fit` et `predict` ?**

`fit` apprend les transformations et le modèle. `predict` utilise cet apprentissage pour produire des classes sans réentraîner.

**Pourquoi `model__C` contient-il deux underscores ?**

`model` est le nom de l'étape du pipeline ; `C` est son hyperparamètre. Cette notation permet à la recherche en grille de modifier le bon réglage dans la bonne étape.

**Que signifie `C` ?**

Il règle l'inverse de la force de régularisation de la régression logistique. Une valeur plus petite pénalise davantage les coefficients. On compare plusieurs valeurs sur la validation.

**Pourquoi standardiser ? Est-ce utile aux trois modèles ?**

Cela met les variables numériques sur des échelles comparables, ce qui est utile à la régression logistique régularisée. Les arbres n'en ont généralement pas besoin ; le projet garde un prétraitement commun pour simplifier la comparaison.

**Que fait `class_weight="balanced"` ?**

Il donne davantage de poids à la classe minoritaire pendant l'entraînement. Il ne crée pas de nouvelles données et ne rend pas les effectifs égaux. Les scores de probabilité ne doivent pas être considérés comme parfaitement calibrés sans vérification.

**Pourquoi `random_state=42` ?**

C'est une graine arbitraire qui stabilise les tirages aléatoires. Elle facilite la reproduction avec les mêmes données et le même environnement, sans garantir des résultats identiques entre toutes les versions des bibliothèques.

**Pourquoi cinq plis ?**

Chaque réglage est évalué sur cinq partitions internes de l'entraînement. Dans ce cas de classification, scikit-learn utilise une répartition stratifiée. Le score moyen guide le choix et le meilleur pipeline est ensuite réentraîné sur tout le jeu d'entraînement.

**Pourquoi la forêt utilise-t-elle `n_jobs=1` dans la recherche ?**

La recherche en grille répartit déjà les essais en parallèle avec `n_jobs=-1`. Garder un seul processus par forêt évite d'empiler deux niveaux de parallélisme.

**À quoi sert `if __name__ == "__main__"` ?**

À lancer `main()` quand on exécute le script directement. Importer ses fonctions dans un notebook ne lance pas l'entraînement automatiquement.

**Pourquoi remplacer certaines fonctions dans le test de cohérence ?**

Le test fournit des clients fictifs, des grilles courtes et un dossier temporaire. L'optimisation et le calcul des métriques sont réellement exécutés. Il vérifie la cohérence du programme sans dépendre du CSV Telco.

## 8. Limites à reconnaître

- Les scores ont été recalculés sur le CSV Telco retrouvé. Ils restent propres à ce jeu de données et à ce découpage.
- Les tests couvrent des cas ciblés, pas toutes les erreurs possibles ni l'exécution complète des graphiques.
- Le test de cohérence retrouve certaines cellules par leur position : si le notebook est réorganisé, ce test doit être adapté.
- Les versions des dépendances ne sont pas entièrement figées. La reproductibilité pourrait être renforcée avec un environnement versionné.
- La commande de visualisation enregistre les versions et l’empreinte du CSV dans un manifeste. Le pipeline entraîné n’est pas encore sauvegardé.
- Le notebook explore différents seuils sur le test. Pour choisir un seuil opérationnel, il faudrait une validation séparée.

Ces limites permettent de présenter honnêtement le travail et d'expliquer les prochaines améliorations.


## 9. Vérification locale effectuée

Le 15 septembre 2026, les sept tests ont réussi dans l'environnement local utilisé pour la correction. La syntaxe des fichiers Python modifiés a également été vérifiée. Cela confirme les comportements décrits dans la section tests, sans valider une exécution distante de GitHub Actions. Une exécution ultérieure de la commande de visualisation a produit les scores Telco et les quatre figures.


Mise à jour du point 3 : neuf tests passent désormais, dont la vérification des coefficients exportés et des effectifs des figures. Les deux notebooks ont également été exécutés entièrement avec les données réelles ; leurs sorties sont enregistrées. Les tableaux du notebook concordent avec ceux de la commande de visualisation.
