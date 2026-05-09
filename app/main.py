"""FastAPI application for Titanic survival prediction."""

import os
import mlflow.sklearn
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse, PredictionResult

# ── Global model store ─────────────────────────────────────────────
MODEL = None
MODEL_NAME = "titanic-random_forest"
MODEL_ALIAS = "production"


def load_model():
    """Load the Production model from MLflow registry."""
    global MODEL
    
    import os
    os.environ["MLFLOW_TRACKING_USERNAME"] = "omar.sameh.shamakh"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN", "")
    
    mlflow.set_tracking_uri(
        "https://dagshub.com/omar.sameh.shamakh/lab0-titanic-pipeline.mlflow"
    )
    
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model: {model_uri}")
    MODEL = mlflow.sklearn.load_model(model_uri)
    print("Model loaded and ready!")


# ── Lifespan: runs on startup and shutdown ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model once
    load_model()
    yield
    # Shutdown: cleanup (nothing needed here)
    print("Shutting down...")


# ── Create the FastAPI app ─────────────────────────────────────────
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict survival for Titanic passengers using a trained Random Forest model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health check endpoint ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Titanic Survival Prediction API",
        "status": "running",
        "model": f"{MODEL_NAME}@{MODEL_ALIAS}",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


# ── Prediction endpoint ────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict survival for one or more passengers."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert request to DataFrame
    df = pd.DataFrame([p.model_dump() for p in request.passengers])

    # Run predictions
    predictions = MODEL.predict(df)
    probabilities = MODEL.predict_proba(df)

    # Build response
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        survived = bool(pred == 1)
        results.append(PredictionResult(
            passenger_index=i,
            survived=survived,
            survival_probability=round(float(prob[1]), 4),
            prediction_label="SURVIVED" if survived else "DID NOT SURVIVE",
        ))

    return PredictResponse(
        model_name=f"{MODEL_NAME}@{MODEL_ALIAS}",
        total_passengers=len(results),
        predictions=results,
    )