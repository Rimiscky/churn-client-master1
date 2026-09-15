"""API locale de démonstration : santé, prédiction et suivi des lots."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from threading import Lock
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.inference import MODEL_PATH, predict_clients
from src.train import PROJECT_ROOT


class PredictionRequest(BaseModel):
    clients: list[dict] = Field(min_length=1, max_length=5000)
    threshold: float = Field(default=.5, ge=0, le=1)


def create_app(model_path=MODEL_PATH, log_path=None):
    """Charger une seule fois le modèle ; protéger les compteurs contre les accès simultanés."""
    log_path = Path(log_path) if log_path else PROJECT_ROOT / 'runtime/monitoring.jsonl'
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = joblib.load(model_path)
        app.state.stats = {'requests': 0, 'failed': 0, 'rows_scored': 0, 'batches_with_alerts': 0, 'last_report': None}
        yield

    app = FastAPI(title='Churn client - démonstration Master 1', lifespan=lifespan)

    @app.middleware('http')
    async def count_prediction_requests(request, call_next):
        response = await call_next(request)
        # Compter aussi les requêtes rejetées avant d'entrer dans la fonction de prédiction.
        if request.url.path == '/predict' and request.method == 'POST':
            with lock:
                app.state.stats['requests'] += 1
                app.state.stats['failed'] += response.status_code >= 400
        return response

    @app.get('/health')
    def health():
        return {'status': 'ok', 'version_modele': app.state.bundle['version']}

    @app.post('/predict')
    def predict(request: PredictionRequest):
        started = time.perf_counter()
        try:
            predictions, report = predict_clients(pd.DataFrame(request.clients), app.state.bundle, request.threshold)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        report['duration_ms'] = round((time.perf_counter()-started)*1000, 2)
        # Le journal contient uniquement des agrégats, jamais les identifiants des clients.
        with lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a') as stream:
                stream.write(json.dumps(report, ensure_ascii=False)+'\n')
            stats = app.state.stats
            stats['rows_scored'] += len(predictions)
            stats['batches_with_alerts'] += bool(report['alerts'])
            stats['last_report'] = report
        return {'predictions': predictions.to_dict('records'), 'monitoring': report}

    @app.get('/monitoring')
    def monitoring():
        with lock:
            return dict(app.state.stats)

    dashboard_dir = PROJECT_ROOT / 'results/dashboard'
    if dashboard_dir.exists():
        app.mount('/dashboard', StaticFiles(directory=dashboard_dir, html=True), name='dashboard')
    return app


app = create_app(Path(os.environ.get('CHURN_MODEL_PATH', str(MODEL_PATH))))
