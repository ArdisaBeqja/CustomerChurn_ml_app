from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.database import PredictionRecord, SessionLocal, init_db
from app.schemas import CustomerFeatures, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer is likely to churn using a trained ML model.",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def startup_event() -> None:
    global model
    init_db()
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python -m ml.train` first.",
        )

    row = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(row)[0][1])
    prediction = probability >= 0.5

    db = SessionLocal()
    try:
        db.add(
            PredictionRecord(
                **customer.model_dump(),
                churn_prediction=prediction,
                churn_probability=probability,
            )
        )
        db.commit()
    finally:
        db.close()

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        model_name=model.named_steps["classifier"].__class__.__name__,
    )
