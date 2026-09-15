"""Prédiction par lots et contrôles de données, sans réentraîner le modèle."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.train import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / 'models/churn.joblib'


def make_bundle(pipeline, X_train, version):
    """Conserver le pipeline et une référence de qualité apprise sur le train uniquement."""
    reference = {}
    for column in X_train:
        values = X_train[column]
        item = {'missing_rate': float(values.isna().mean())}
        if pd.api.types.is_numeric_dtype(values):
            item.update(kind='numeric', median=float(values.median()),
                        q05=float(values.quantile(.05)), q95=float(values.quantile(.95)))
        else:
            item.update(kind='categorical', categories=sorted(values.dropna().unique().tolist()))
        reference[column] = item
    return {'pipeline': pipeline, 'version': version, 'columns': list(X_train.columns),
            'reference': reference, 'reference_mean_score': float(pipeline.predict_proba(X_train)[:, 1].mean())}


def prepare_clients(frame, bundle):
    """Valider les entrées et garder l'ordre des clients, même si leurs profils sont identiques."""
    if frame.empty:
        raise ValueError('Le fichier ne contient aucun client.')
    required = ['customerID'] + bundle['columns']
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError('Colonnes requises absentes : ' + ', '.join(missing))
    extra = sorted(set(frame.columns) - set(required))
    if extra:
        raise ValueError('Colonnes inattendues (la cible Churn est interdite) : ' + ', '.join(extra))
    if frame.customerID.isna().any() or frame.customerID.astype(str).str.strip().eq('').any():
        raise ValueError('Chaque client doit avoir un identifiant renseigné.')
    ids = frame.customerID.astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError('Les identifiants clients doivent être uniques dans un lot.')
    X = frame[bundle['columns']].copy()
    quality = {}
    alerts = []
    for column, ref in bundle['reference'].items():
        values = X[column].replace(r'^\s*$', np.nan, regex=True)
        if ref['kind'] == 'numeric':
            original = values.notna()
            values = pd.to_numeric(values, errors='coerce')
            invalid = int((original & values.isna()).sum())
            if np.isinf(values).any() or (values.dropna() < 0).any():
                raise ValueError(f'{column} contient une valeur infinie ou négative.')
            if column == 'SeniorCitizen' and not values.dropna().isin([0, 1]).all():
                raise ValueError('SeniorCitizen doit contenir 0 ou 1.')
            # Un décalage de médiane est un signal exploratoire, pas une preuve de dérive nuisible.
            spread = max(ref['q95'] - ref['q05'], 1.)
            median_shift = abs(float(values.median()) - ref['median']) / spread if values.notna().any() else None
            item = {'invalid_count': invalid, 'median_shift_scaled': median_shift}
            if invalid:
                alerts.append(f'{column} : {invalid} valeur(s) invalide(s) convertie(s) en valeur manquante.')
            if median_shift is not None and median_shift > .5:
                alerts.append(f'{column} : médiane éloignée de la référence d’entraînement.')
        else:
            if not values.dropna().map(lambda value: isinstance(value, str)).all():
                raise ValueError(f'{column} attend des catégories textuelles.')
            unknown = values.notna() & ~values.isin(ref['categories'])
            item = {'unknown_count': int(unknown.sum()), 'unknown_rate': float(unknown.mean())}
            if unknown.any():
                alerts.append(f'{column} : {int(unknown.sum())} catégorie(s) inconnue(s).')
        item['missing_rate'] = float(values.isna().mean())
        if item['missing_rate'] > ref['missing_rate'] + .05:
            alerts.append(f'{column} : valeurs manquantes en hausse de plus de 5 points.')
        quality[column] = item
        X[column] = values
    return ids, X, quality, alerts


def predict_clients(frame, bundle, threshold=.5):
    """Calculer le score et produire un rapport agrégé, sans écrire de données individuelles au journal."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Le seuil doit être compris entre 0 et 1.')
    ids, X, quality, alerts = prepare_clients(frame, bundle)
    scores = bundle['pipeline'].predict_proba(X)[:, 1]
    if abs(float(scores.mean()) - bundle['reference_mean_score']) > .10:
        alerts.append('Le score moyen diffère de plus de 10 points de la référence d’entraînement.')
    now = datetime.now(timezone.utc).isoformat()
    predictions = pd.DataFrame({'customerID': ids.to_numpy(), 'score_churn': scores,
        'a_examiner': scores >= threshold, 'rang_priorite': pd.Series(scores).rank(method='first', ascending=False).astype(int),
        'date_score': now, 'version_modele': bundle['version']})
    report = {'timestamp_utc': now, 'version_modele': bundle['version'], 'rows': len(X),
        'threshold': threshold, 'flagged_count': int((scores >= threshold).sum()),
        'mean_score': float(scores.mean()), 'quality': quality, 'alerts': alerts,
        'performance': None, 'note': 'Les performances ne sont pas calculables sans départs réellement observés. Seuils d’alerte pédagogiques, à ajuster.'}
    return predictions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'runtime/predictions.csv')
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    # Charger uniquement un artefact produit par ce projet et provenant d'une source de confiance.
    bundle = joblib.load(args.model)
    predictions, report = predict_clients(pd.read_csv(args.input), bundle, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    args.output.with_suffix('.monitoring.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'{len(predictions)} clients traités ; {len(report["alerts"])} alerte(s). Résultats : {args.output}')


if __name__ == '__main__':
    main()
