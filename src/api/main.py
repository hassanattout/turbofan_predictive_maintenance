from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import (
    BASE_FEATURES,
    ROLLING_WINDOW,
    latest_feature_row,
    maintenance_decision,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "rf_model.pkl"

app = FastAPI(
    title="Turbofan RUL Prediction API",
    version="2.0.0",
    description=(
        "Predict RUL from an ordered sensor history. "
        "At least five cycles are required because the model uses "
        "rolling and trend features."
    ),
)
model = joblib.load(MODEL_PATH)


class SensorReading(BaseModel):
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    readings: list[SensorReading] = Field(
        min_length=ROLLING_WINDOW,
        max_length=500,
        description="Ordered oldest-to-newest engine observations.",
    )


@app.get("/")
def home():
    return {
        "status": "API is running",
        "model_input": f"At least {ROLLING_WINDOW} ordered cycles",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH.name),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    history = pd.DataFrame(
        [reading.model_dump() for reading in request.readings],
        columns=BASE_FEATURES,
    )
    try:
        features = latest_feature_row(history)
        prediction = float(model.predict(features)[0])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "predicted_rul": round(prediction, 2),
        "decision": maintenance_decision(prediction),
        "cycles_used": len(history),
    }
