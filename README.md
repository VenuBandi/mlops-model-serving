# MLOps Model Serving

Production-ready ML model serving with MLflow experiment tracking and a FastAPI REST endpoint. Demonstrates the full MLOps lifecycle: train → log → register → serve → monitor.

## Architecture

```
train/train_and_log.py
    └── XGBoost training
        └── mlflow.log_params / log_metrics / log_model
            └── MLflow Model Registry (xgb_demo/latest)
                └── api/main.py (FastAPI)
                    └── /predict endpoint
                    └── /health endpoint
                        └── Docker container
```

## Quickstart

```bash
pip install -r requirements.txt

# Train and log to MLflow
python train/train_and_log.py

# Start MLflow UI
mlflow ui --port 5000
# → open http://localhost:5000

# Serve the model
MODEL_URI="models:/xgb_demo/latest" uvicorn api.main:app --reload --port 8000
```

## API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_0": 1.2, "feature_1": -0.5, ...}}'

# {"prediction": 1, "probability": 0.823, "model_uri": "models:/xgb_demo/latest"}
```

## Docker

```bash
docker build -t mlops-serving .
docker run -p 8000:8000 -e MODEL_URI="models:/xgb_demo/latest" mlops-serving
```

## Stack

Python · XGBoost · MLflow · FastAPI · Docker · scikit-learn · pandas
