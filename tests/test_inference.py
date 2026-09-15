"""Tester le parcours réel de prédiction et de suivi avec des clients fictifs."""
import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fastapi.testclient import TestClient
from src.train import build_preprocessor
from src.inference import make_bundle,predict_clients
from src.monitor import evaluate_observed
from src.api import create_app


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.X=pd.DataFrame({'tenure':[1,2,3,10,12,15], 'Contract':['A','B','A','B','A','B'], 'TotalCharges':[10.,20.,30.,100.,120.,150.]})
        self.y=pd.Series([1,1,1,0,0,0])
        pipeline=Pipeline([('prep',build_preprocessor(self.X)),('model',LogisticRegression())]).fit(self.X,self.y)
        self.bundle=make_bundle(pipeline,self.X,'test-v1')
        self.clients=self.X.assign(customerID=[f'C-{i}' for i in range(6)])

    def test_saved_pipeline_preserves_predictions_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.joblib';joblib.dump(self.bundle,path)
            loaded=joblib.load(path)
            predictions,report=predict_clients(self.clients.iloc[::-1],loaded)
            np.testing.assert_allclose(predictions.score_churn,loaded['pipeline'].predict_proba(self.X.iloc[::-1])[:,1])
            self.assertEqual(predictions.customerID.tolist(),self.clients.customerID.iloc[::-1].tolist())
            self.assertIsNone(report['performance'])
            self.assertEqual(report['version_modele'],'test-v1')

    def test_missing_columns_and_target_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Colonnes requises'):
            predict_clients(self.clients.drop(columns='Contract'),self.bundle)
        with self.assertRaisesRegex(ValueError,'Churn'):
            predict_clients(self.clients.assign(Churn='Yes'),self.bundle)

    def test_duplicate_ids_and_invalid_numbers_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'uniques'):
            predict_clients(self.clients.assign(customerID='same'),self.bundle)
        with self.assertRaisesRegex(ValueError,'infinie ou négative'):
            predict_clients(self.clients.assign(tenure=np.inf),self.bundle)
        with self.assertRaisesRegex(ValueError,'seuil'):
            predict_clients(self.clients,self.bundle,float('nan'))

    def test_altered_data_raise_quality_alerts(self):
        altered=self.clients.assign(Contract='NEW',TotalCharges=np.nan)
        predictions,report=predict_clients(altered,self.bundle)
        self.assertEqual(len(predictions),6)
        self.assertEqual(report['quality']['Contract']['unknown_count'],6)
        self.assertEqual(report['quality']['TotalCharges']['missing_rate'],1.)
        self.assertTrue(report['alerts'])

    def test_observed_results_are_joined_by_id(self):
        predictions,_=predict_clients(self.clients,self.bundle)
        outcomes=pd.DataFrame({'customerID':self.clients.customerID,'Churn':self.y.map({0:'No',1:'Yes'})})
        result=evaluate_observed(predictions,outcomes.iloc[::-1])
        self.assertEqual(result['rows'],6)
        self.assertEqual(result['observed_churn'],3)
        self.assertEqual(sum(map(sum,result['confusion_matrix'])),6)
        with self.assertRaisesRegex(ValueError,'exactement'):
            evaluate_observed(predictions,outcomes.iloc[:-1])

    def test_api_predict_monitoring_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.joblib';joblib.dump(self.bundle,path)
            log=Path(d)/'monitoring.jsonl'
            with TestClient(create_app(path,log)) as client:
                self.assertEqual(client.get('/health').json()['status'],'ok')
                response=client.post('/predict',json={'clients':self.clients.to_dict('records')})
                self.assertEqual(response.status_code,200)
                self.assertEqual(len(response.json()['predictions']),6)
                self.assertEqual(client.get('/monitoring').json()['rows_scored'],6)
                invalid=client.post('/predict',json={'clients':self.clients.drop(columns='Contract').to_dict('records')})
                self.assertEqual(invalid.status_code,422)
                self.assertEqual(client.get('/monitoring').json()['failed'],1)
                self.assertEqual(client.post('/predict',json={'clients':[]}).status_code,422)
                self.assertEqual(client.get('/monitoring').json()['failed'],2)
            saved=json.loads(log.read_text().splitlines()[0])
            self.assertEqual(saved['rows'],6)
            self.assertNotIn('customerID',log.read_text())


if __name__=='__main__':
    unittest.main()
