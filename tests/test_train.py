"""Vérifications sur des données fictives, sans utiliser le CSV privé du projet."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import train


class TrainingTests(unittest.TestCase):
    def test_missing_file_has_clear_error(self):
        # Un fichier absent doit produire une explication compréhensible.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Dataset introuvable"):
                train.load_data(Path(directory) / "absent.csv")

    def test_missing_target_has_clear_error(self):
        # Un CSV lisible ne suffit pas : la colonne à prédire doit être présente.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sans_cible.csv"
            pd.DataFrame({"customerID": ["A"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "colonne cible 'Churn'"):
                train.load_data(path)

    def test_distinct_clients_are_preserved(self):
        # Deux identifiants différents restent deux clients, même avec le même profil.
        raw = pd.DataFrame({
            "customerID": ["A", "B", "A"],
            "TotalCharges": [" ", " ", " "],
            "Churn": ["Yes", "Yes", "Yes"],
        })
        cleaned = train.clean_data(raw)
        self.assertEqual(cleaned["customerID"].tolist(), ["A", "B"])
        self.assertTrue(cleaned["TotalCharges"].isna().all())
        X, y = train.split_features_target(cleaned)
        self.assertNotIn("customerID", X.columns)
        self.assertEqual(y.tolist(), [1, 1])
        self.assertEqual(len(raw), 3)

    def test_profiles_without_identifiers_are_preserved(self):
        # Sans identifiant, le nettoyage ne peut pas conclure que deux profils sont un doublon.
        raw = pd.DataFrame({"TotalCharges": ["12", "12"], "Churn": ["No", "No"]})
        self.assertEqual(len(train.clean_data(raw)), 2)

    def test_invalid_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cible Churn"):
            train.split_features_target(pd.DataFrame({"Churn": ["Yes", "inconnu"]}))

    def test_preprocessing_handles_missing_and_unknown_values(self):
        # Une catégorie nouvelle dans le test ne doit pas faire échouer la prédiction.
        X = pd.DataFrame({"TotalCharges": [10., 20., np.nan, 30.], "Contract": ["A", "B", "A", "B"]})
        pipe = Pipeline([("prep", train.build_preprocessor(X)), ("model", LogisticRegression())])
        pipe.fit(X, [0, 1, 0, 1])
        prediction = pipe.predict_proba(pd.DataFrame({"TotalCharges": [np.nan], "Contract": ["nouveau"]}))
        self.assertTrue(np.isfinite(prediction).all())
        # La médiane doit provenir de l'entraînement et rester inchangée après prédiction.
        self.assertEqual(pipe.named_steps["prep"].named_transformers_["num"].named_steps["imputer"].statistics_[0], 20.)

    def test_script_and_notebook_export_same_results(self):
        # Comparer les vrais parcours sur un petit jeu fictif et des grilles réduites.
        rng = np.random.default_rng(42)
        raw = pd.DataFrame({
            "customerID": [f"client-{i}" for i in range(60)],
            "TotalCharges": rng.uniform(1, 100, 60).astype(str),
            "Contract": ["A", "B", "C"] * 20,
            "Churn": ["No", "Yes"] * 30,
        })
        searches = train.get_searches()
        for model, params in searches.values():
            for key, values in params.items():
                params[key] = [values[0]]
            if "model__n_estimators" in params:
                params["model__n_estimators"] = [5]
        notebook = json.loads((train.PROJECT_ROOT / "notebooks/02_modelisation_evaluation.ipynb").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original_save = train.save_results
            def save_in_temporary_directory(scores, selected):
                return original_save(scores, selected, output)

            with patch.object(train, "load_data", return_value=raw), patch.object(train, "get_searches", return_value=searches), patch.object(train, "save_results", side_effect=save_in_temporary_directory), contextlib.redirect_stdout(io.StringIO()):
                train.main()
                script_scores = pd.read_csv(output / "metrics.csv")
                script_params = json.loads((output / "best_parameters.json").read_text())
                X, y = train.split_features_target(train.clean_data(raw))
                namespace = {
                    "X": X, "y": y, "pd": pd, "train_test_split": train_test_split,
                    "RANDOM_STATE": train.RANDOM_STATE, "TEST_SIZE": train.TEST_SIZE,
                    "optimize_models": train.optimize_models, "evaluate_model": train.evaluate_model,
                    "save_results": train.save_results,
                }
                # Exécuter les cellules réelles de séparation, optimisation et export.
                for index in [3, 17, 19]:
                    exec("".join(notebook["cells"][index]["source"]), namespace)
                pd.testing.assert_frame_equal(script_scores, pd.read_csv(output / "metrics.csv"))
                self.assertEqual(script_params, json.loads((output / "best_parameters.json").read_text()))


if __name__ == "__main__":
    unittest.main()
