"""Créer les quatre figures de soutenance avec les modèles et données du projet.

Depuis la racine : python -m src.visualize
Les figures utilisent les mêmes pipelines et réglages que le script d'entraînement.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
# Un rendu sans fenêtre permet aussi la génération dans les tests et sur un serveur.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.train import (
    DATA_PATH, RESULTS_DIR, RANDOM_STATE, TEST_SIZE, load_data, clean_data,
    split_features_target, optimize_models, evaluate_model, save_results,
)

BLUE = "#2463A5"
ORANGE = "#C75C17"
INK = "#183247"
GREY = "#718096"
BACKGROUND = "#F7F9FC"
LOGISTIC = "Régression logistique"


def french_number(value: float, digits: int = 1) -> str:
    """Afficher les décimales avec une virgule, sans dépendre de la langue du système."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def new_figure(title: str, subtitle: str):
    """Préparer un format 16:9 et des caractères lisibles sur une diapositive."""
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BACKGROUND)
    fig.text(0.055, 0.935, title, fontsize=23, weight="bold", color=INK)
    fig.text(0.055, 0.88, subtitle, fontsize=12, color=GREY)
    return fig


def finish_figure(fig, output_dir: Path, stem: str, note: str):
    """Exporter chaque graphique en image et en PDF vectoriel, puis libérer la mémoire."""
    fig.text(0.055, 0.04, note, fontsize=10, color=GREY)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def style_axis(ax):
    ax.set_facecolor(BACKGROUND)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D5DEE8")
    ax.tick_params(colors=INK, labelsize=12, length=0, pad=8)
    ax.set_axisbelow(True)


def feature_label(name: str) -> str:
    """Traduire les noms issus du prétraitement sans perdre leur modalité."""
    name = name.split("__", 1)[-1]
    variables = {
        "tenure": "Ancienneté", "MonthlyCharges": "Frais mensuels",
        "TotalCharges": "Frais totaux", "SeniorCitizen": "Client senior",
        "Contract": "Contrat", "InternetService": "Internet",
        "PaymentMethod": "Paiement", "PaperlessBilling": "Facture dématérialisée",
        "OnlineSecurity": "Sécurité en ligne", "TechSupport": "Support technique",
        "OnlineBackup": "Sauvegarde en ligne", "DeviceProtection": "Protection appareil",
        "StreamingTV": "TV en streaming", "StreamingMovies": "Films en streaming",
        "PhoneService": "Téléphone", "MultipleLines": "Lignes multiples",
        "Partner": "Conjoint", "Dependents": "Personnes à charge", "gender": "Genre",
    }
    modalities = {
        "Yes": "oui", "No": "non", "Month-to-month": "mensuel", "One year": "1 an",
        "Two year": "2 ans", "Fiber optic": "fibre optique", "DSL": "DSL",
        "No internet service": "sans Internet", "No phone service": "sans téléphone",
        "Electronic check": "chèque électronique", "Mailed check": "chèque postal",
        "Bank transfer (automatic)": "virement automatique",
        "Credit card (automatic)": "carte automatique", "Female": "femme", "Male": "homme",
    }
    if "_" not in name:
        return variables.get(name, name) + " (standardisé)"
    variable, value = name.split("_", 1)
    return f"{variables.get(variable, variable)} : {modalities.get(value, value)}"


def coefficient_table(pipeline) -> pd.DataFrame:
    """Associer chaque coefficient au bon nom de colonne après encodage."""
    prep = pipeline.named_steps["prep"]
    coefficients = pipeline.named_steps["model"].coef_[0]
    return pd.DataFrame({
        "variable": prep.get_feature_names_out(),
        "libelle": [feature_label(name) for name in prep.get_feature_names_out()],
        "coefficient": coefficients,
    }).sort_values("coefficient").reset_index(drop=True)


def export_visuals(y_all, X_test, y_test, best_models, scores, output_dir=None):
    """Dessiner à partir de modèles déjà entraînés, sans relancer leur apprentissage."""
    output_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    # 1. Effectifs réels, après nettoyage. Une barre part de zéro pour rester honnête.
    counts = pd.Series(y_all).value_counts().reindex([0, 1], fill_value=0)
    shares = counts / counts.sum()
    pd.DataFrame({"classe": ["Reste", "Part"], "effectif": counts.values, "part": shares.values}).to_csv(output_dir / "01_repartition.csv", index=False)
    fig = new_figure(
        f"{french_number(shares[1] * 100)} % des clients ont quitté l'entreprise",
        f"Répartition de la cible après nettoyage | {french_number(counts.sum(), 0)} clients",
    )
    ax = fig.add_axes([0.19, 0.24, 0.73, 0.52])
    style_axis(ax)
    ax.barh([1, 0], shares.values, height=0.48, color=[BLUE, ORANGE])
    ax.set_yticks([1, 0], ["Reste", "Part (churn)"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Part des clients", fontsize=12, color=INK)
    ax.grid(axis="x", alpha=0.15)
    for pos, count, share in zip([1, 0], counts.values, shares.values):
        ax.text(share + 0.015, pos, f"{french_number(share * 100)} %\n{french_number(count, 0)} clients", va="center", fontsize=15, color=INK)
    finish_figure(fig, output_dir, "01_repartition_churn", "À retenir : prédire toujours « reste » donne une accuracy élevée, mais ne détecte aucun départ.")

    # 2. Même jeu de test et même échelle pour tous les modèles, baseline incluse.
    comparison = pd.DataFrame(scores).copy()
    comparison.to_csv(output_dir / "02_comparaison.csv", index=False)
    fig = new_figure("Détecter les départs : comparaison des modèles", f"Modèles optimisés et référence naïve | Test : {len(y_test):,} clients".replace(",", " "))
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.23, right=0.96, top=0.77, bottom=0.23, wspace=0.13)
    for ax, metric, label in zip(axes, ["accuracy", "recall", "roc_auc"], ["Accuracy", "Rappel des départs", "ROC-AUC"]):
        style_axis(ax)
        colors = [ORANGE if name == LOGISTIC else GREY if name == "Référence naïve" else BLUE for name in comparison.model]
        ax.barh(np.arange(len(comparison)), comparison[metric], color=colors, height=0.52)
        ax.set_yticks(np.arange(len(comparison)), comparison.model)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0,5", "1"])
        ax.set_title(label, fontsize=14, color=INK, pad=16)
        ax.grid(axis="x", alpha=0.15)
        for pos, value in enumerate(comparison[metric]):
            ax.text(value - 0.03 if value > 0.25 else value + 0.03, pos, french_number(value, 3), va="center", ha="right" if value > 0.25 else "left", color="white" if value > 0.25 else INK, weight="bold", fontsize=12)
    axes[0].invert_yaxis()
    finish_figure(fig, output_dir, "02_comparaison_modeles", "Orange : régression logistique privilégiée pour l'oral. Petits écarts de scores : aucune significativité établie.")

    # 3. La matrice correspond au modèle optimisé, au seuil par défaut de predict.
    logistic = best_models[LOGISTIC]
    matrix = confusion_matrix(y_test, logistic.predict(X_test), labels=[0, 1])
    pd.DataFrame(matrix, index=["Réel : reste", "Réel : part"], columns=["Prédit : reste", "Prédit : part"]).to_csv(output_dir / "03_matrice_confusion.csv")
    tn, fp, fn, tp = matrix.ravel()
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    fig = new_figure(f"{tp} départs détectés, {fn} départs manqués", "Régression logistique optimisée | Jeu de test | Décision par défaut, seuil voisin de 0,50")
    ax = fig.add_axes([0.12, 0.20, 0.45, 0.59])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    ax.set_xticks([0, 1], ["Reste", "Part"])
    ax.set_yticks([0, 1], ["Reste", "Part"])
    ax.tick_params(labelsize=13, length=0, pad=8)
    ax.set_xlabel("Prédiction du modèle", fontsize=12, labelpad=10)
    ax.set_ylabel("Situation réelle", fontsize=12, labelpad=10)
    labels = [["Vrais négatifs", "Fausses alertes"], ["Départs manqués", "Départs détectés"]]
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]}\n{labels[row][col]}", ha="center", va="center", fontsize=14, color="white" if matrix[row,col] > matrix.max() * .5 else INK)
    fig.text(0.64, 0.68, f"{french_number(recall * 100)} %", fontsize=34, weight="bold", color=ORANGE)
    fig.text(0.64, 0.62, "des départs réels détectés\n(rappel)", fontsize=14, color=INK)
    fig.text(0.64, 0.45, f"{french_number(precision * 100)} %", fontsize=34, weight="bold", color=BLUE)
    fig.text(0.64, 0.39, "des alertes correspondent\nà un départ (précision)", fontsize=14, color=INK)
    finish_figure(fig, output_dir, "03_matrice_confusion", "Le coût d'un départ manqué et celui d'une fausse alerte doivent guider le futur choix du seuil.")

    # 4. Afficher les coefficients signés, pas une fausse importance causale.
    coefficients = coefficient_table(logistic)
    coefficients.to_csv(output_dir / "04_coefficients_complets.csv", index=False)
    selected = coefficients.loc[coefficients.coefficient.abs().nlargest(12).index].sort_values("coefficient")
    fig = new_figure("Les caractéristiques qui contribuent au score", "Régression logistique optimisée | 12 coefficients de plus grande valeur absolue")
    ax = fig.add_axes([0.36, 0.23, 0.58, 0.56])
    style_axis(ax)
    ax.barh(selected.libelle, selected.coefficient, color=[ORANGE if value > 0 else BLUE for value in selected.coefficient], height=0.65)
    limit = max(selected.coefficient.abs().max(), 0.1) * 1.28
    ax.set_xlim(-limit, limit)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.grid(axis="x", alpha=0.15)
    ax.set_xlabel("Coefficient : négatif = score réduit ; positif = score accru", fontsize=11, labelpad=12)
    for pos, value in enumerate(selected.coefficient):
        ax.text(value + (0.035 if value >= 0 else -0.035) * limit, pos, french_number(value, 2), va="center", ha="left" if value >= 0 else "right", fontsize=11, color=INK)
    finish_figure(fig, output_dir, "04_coefficients_logistique", "Associations, pas causes. Variables numériques standardisées ; toutes les modalités sont encodées.\nCoefficients sensibles aux variables corrélées ; une modalité ne se lit pas comme un effet face à une référence omise.")
    return [output_dir / f"{stem}.png" for stem in ["01_repartition_churn", "02_comparaison_modeles", "03_matrice_confusion", "04_coefficients_logistique"]]


def main():
    """Recalculer les modèles, puis produire les figures et leur trace de provenance."""
    raw = load_data()
    df = clean_data(raw)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print("Optimisation des trois modèles sur cinq plis...", flush=True)
    models, searches = optimize_models(X_train, y_train)
    scores = [evaluate_model(name, model, X_test, y_test) for name, model in models.items()]
    save_results(scores, searches)
    # La référence naïve est entraînée sur le train, jamais choisie à partir des classes du test.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline = evaluate_model("Référence naïve", dummy, X_test, y_test)
    paths = export_visuals(y, X_test, y_test, models, scores + [baseline])
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATA_PATH.name,
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "clean_rows": len(df), "train_rows": len(X_train), "test_rows": len(X_test),
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "cv": 5,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "searches": searches,
    }
    (RESULTS_DIR / "visualisation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for path in paths:
        print(f"Figure créée : {path}", flush=True)


if __name__ == "__main__":
    main()
