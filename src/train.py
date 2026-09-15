from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lire le CSV et signaler clairement un fichier ou une cible absents."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}. "
            "Télécharge le fichier Telco Customer Churn depuis Kaggle et place-le dans data/raw/."
        )

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET}' est absente du dataset.")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyer une copie du tableau, sans apprendre de statistique sur les données."""
    # La copie garde le tableau d'origine intact pour l'analyse exploratoire.
    df = df.copy()

    # Comparer les lignes brutes, identifiant inclus, pour conserver les clients distincts.
    # Sans identifiant, des profils identiques ne prouvent pas qu'il s'agit de doublons.
    if "customerID" in df.columns:
        df = df.drop_duplicates().reset_index(drop=True)

    # Dans le dataset Telco, TotalCharges peut contenir des chaînes vides.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def split_features_target(df: pd.DataFrame):
    """Retourner X (caractéristiques) et y (cible binaire), sans l'identifiant client."""
    # Refuser une cible inattendue évite de transformer une erreur de saisie en classe valide.
    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("La cible Churn contient des valeurs autres que 'Yes' et 'No'.")

    X = df.drop(columns=[TARGET, "customerID"], errors="ignore")
    return X, y.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Définir les transformations numériques et catégorielles, sans les entraîner."""
    # Les types de colonnes déterminent quel traitement appliquer à chaque variable.
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # La médiane et l'échelle seront apprises lors de fit, sur l'entraînement uniquement.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Une catégorie jamais vue à l'entraînement ne bloque pas les futures prédictions.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def get_models():
    """Modèles de départ, avant la recherche d'hyperparamètres du notebook."""
    # balanced augmente le poids de la classe minoritaire sans créer de nouveaux clients.
    return {
        "Régression logistique": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Arbre de décision": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def get_searches():
    """Grilles communes au script et au notebook de modélisation."""
    return {
        "Régression logistique": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.1, 1, 10]},
        ),
        "Arbre de décision": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__max_depth": [3, 5, 7], "model__min_samples_leaf": [10, 20, 40]},
        ),
        "Random Forest": (
            # GridSearchCV parallélise déjà les essais : chaque forêt utilise un seul processus.
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [8, None],
                "model__min_samples_leaf": [1, 3],
            },
        ),
    }


def optimize_models(X_train, y_train):
    """Sélectionner les réglages uniquement sur l'entraînement, avec cinq plis."""
    best_models = {}
    search_results = []
    for name, (model, params) in get_searches().items():
        # Créer un prétraitement indépendant pour ne pas partager un objet déjà entraîné.
        pipeline = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])
        # GridSearchCV réapprend le pipeline dans chaque pli, sans consulter le test.
        grid = GridSearchCV(pipeline, params, scoring="roc_auc", cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        search_results.append({
            "model": name,
            "roc_auc_cv": float(grid.best_score_),
            "best_params": grid.best_params_,
        })
    return best_models, search_results


def save_results(scores, searches, output_dir: Path = RESULTS_DIR):
    """Exporter les scores et les paramètres issus de la même exécution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(scores).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    table.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "best_parameters.json").write_text(
        json.dumps(searches, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return table


def evaluate_model(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    """Mesurer un modèle déjà entraîné, sans le réentraîner sur le jeu de test."""
    # Les classes servent à calculer précision/rappel ; les scores servent au ROC-AUC.
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    scores = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n=== {name} ===")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{metric:>10}: {scores[metric]:.3f}")

    print("Matrice de confusion:")
    print(confusion_matrix(y_test, y_pred))

    return scores


def main() -> None:
    """Enchaîner le chargement, l'optimisation, l'évaluation et l'export des résultats."""
    df = clean_data(load_data())
    X, y = split_features_target(df)

    print(f"Lignes après nettoyage : {len(df)}")
    print(f"Taux de churn : {y.mean():.1%}")

    # stratify conserve approximativement la proportion de churn dans les deux groupes.
    # La graine fixe le tirage aléatoire pour faciliter la comparaison des exécutions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Évaluer les modèles optimisés, comme dans la comparaison finale du notebook.
    best_models, searches = optimize_models(X_train, y_train)
    for result in searches:
        print(f"{result['model']} : {result['best_params']} (ROC-AUC CV : {result['roc_auc_cv']:.3f})")
    results = [
        evaluate_model(name, pipeline, X_test, y_test)
        for name, pipeline in best_models.items()
    ]
    results_df = save_results(results, searches)

    print("\n=== Comparaison finale ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nRésultats enregistrés dans : {RESULTS_DIR}")


if __name__ == "__main__":
    # Importer ce fichier depuis un notebook ne lance pas automatiquement l'entraînement.
    main()
