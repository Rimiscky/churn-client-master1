"""Vérifier la fidélité des tableaux qui alimentent les figures de soutenance."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.train import build_preprocessor, evaluate_model
from src.visualize import coefficient_table, export_visuals, LOGISTIC


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        # Des noms et catégories différents permettent de détecter un mauvais alignement.
        self.X = pd.DataFrame({"tenure": [1, 2, 3, 10, 12, 15], "Contract": ["B", "A", "B", "A", "B", "A"]})
        self.y = pd.Series([1, 1, 1, 0, 0, 0])
        self.model = Pipeline([("prep", build_preprocessor(self.X)), ("model", LogisticRegression())]).fit(self.X, self.y)

    def test_named_coefficients_reconstruct_model_score(self):
        # Recalculer le score à partir des noms exportés : un mauvais ordre ferait échouer ce test.
        prep = self.model.named_steps["prep"]
        names = prep.get_feature_names_out()
        coefficients = coefficient_table(self.model).set_index("variable").loc[names, "coefficient"].to_numpy()
        reconstructed = prep.transform(self.X) @ coefficients + self.model.named_steps["model"].intercept_[0]
        np.testing.assert_allclose(reconstructed, self.model.decision_function(self.X))

    def test_exported_counts_match_predictions(self):
        scores = [evaluate_model(LOGISTIC, self.model, self.X, self.y)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = export_visuals(self.y, self.X, self.y, {LOGISTIC: self.model}, scores, output)
            self.assertEqual(len(paths), 4)
            for path in paths:
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
                self.assertTrue(path.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
            distribution = pd.read_csv(output / "01_repartition.csv")
            self.assertEqual(distribution.effectif.sum(), len(self.y))
            matrix = pd.read_csv(output / "03_matrice_confusion.csv", index_col=0).to_numpy()
            predicted = self.model.predict(self.X)
            self.assertEqual(matrix.sum(), len(self.y))
            self.assertEqual(matrix[1, 1], ((self.y == 1) & (predicted == 1)).sum())
            self.assertEqual(matrix[0, 1], ((self.y == 0) & (predicted == 1)).sum())


if __name__ == "__main__":
    unittest.main()
