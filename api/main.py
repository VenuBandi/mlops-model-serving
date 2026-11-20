"""FastAPI model serving endpoint backed by MLflow model registry."""
import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="ML Model Serving API", version="1.0.0")

MODEL_URI = os.getenv("MODEL_URI", "models:/xgb_demo/latest")
_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Loading model from {MODEL_URI}...")
        _model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Model loaded.")
    return _model


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        example={f"feature_{i}": 0.0 for i in range(20)},
    )

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_uri: str


@app.on_event("startup")
def preload():
    get_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = get_model()
        X = pd.DataFrame([req.features])
        proba = float(model.predict(X)[0])
        return PredictResponse(
            prediction=int(proba >= 0.5),
            probability=round(proba, 6),
            model_uri=MODEL_URI,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
