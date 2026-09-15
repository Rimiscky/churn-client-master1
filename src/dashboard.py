"""Construire un dashboard Plotly autonome à partir des seules prédictions du test."""
import json
from pathlib import Path
from plotly.offline import get_plotlyjs
from src.train import PROJECT_ROOT


def build_dashboard(X_test, y_test, models, output_dir):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    scores={name:model.predict_proba(X_test)[:,1] for name,model in models.items()}
    records=[]
    for i,(_,row) in enumerate(X_test.iterrows()):
        records.append({'client':f'Client test {i+1:04d}','contract':row['Contract'],
            'internet':row['InternetService'],'tenure':int(row['tenure']),
            'actual':int(y_test.iloc[i]),'scores':[float(scores[name][i]) for name in models]})
    template=(PROJECT_ROOT/'src/templates/dashboard.html').read_text()
    # Neutraliser une éventuelle fermeture de balise dans une valeur textuelle du jeu de données.
    data=json.dumps(records,ensure_ascii=False).replace('<','\\u003c')
    page=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',data).replace('__MODELS__',json.dumps(list(models),ensure_ascii=False))
    (output_dir/'index.html').write_text(page)
    (output_dir/'predictions_test.json').write_text(json.dumps(records,ensure_ascii=False))
    return output_dir/'index.html'
