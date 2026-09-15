"""Évaluer des prédictions sauvegardées lorsque les départs réels sont enfin connus."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_observed(predictions, outcomes):
    """Joindre explicitement sur l'identifiant, jamais sur l'ordre des lignes."""
    for frame in (predictions, outcomes):
        if 'customerID' not in frame or frame.customerID.isna().any() or frame.customerID.duplicated().any():
            raise ValueError('Identifiants manquants ou répétés.')
    if 'Churn' not in outcomes or not outcomes.Churn.isin(['Yes', 'No']).all():
        raise ValueError('Les résultats observés doivent contenir Churn en Yes/No.')
    if not set(predictions.customerID) == set(outcomes.customerID):
        raise ValueError('Les résultats doivent couvrir exactement le lot de prédictions.')
    if not {'score_churn','a_examiner'}.issubset(predictions.columns):
        raise ValueError('Scores ou décisions absents.')
    joined = predictions.merge(outcomes[['customerID','Churn']], on='customerID', validate='one_to_one')
    y = joined.Churn.map({'No':0,'Yes':1})
    scores = pd.to_numeric(joined.score_churn, errors='coerce')
    if not np.isfinite(scores).all() or not scores.between(0,1).all():
        raise ValueError('Scores invalides.')
    if not joined.a_examiner.isin([True,False,0,1]).all():
        raise ValueError('Décisions invalides.')
    pred = joined.a_examiner.astype(int)
    return {'rows': len(joined), 'observed_churn': int(y.sum()),
        'accuracy': accuracy_score(y,pred), 'precision': precision_score(y,pred,zero_division=0),
        'recall': recall_score(y,pred,zero_division=0), 'f1': f1_score(y,pred,zero_division=0),
        'roc_auc': roc_auc_score(y,scores) if y.nunique()==2 else None,
        'confusion_matrix': confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'note': 'N’utiliser que des résultats dont la période d’observation est complète.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',type=Path)
    parser.add_argument('outcomes',type=Path)
    parser.add_argument('--output',type=Path,default=Path('runtime/performance.json'))
    args=parser.parse_args()
    report=evaluate_observed(pd.read_csv(args.predictions),pd.read_csv(args.outcomes))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
