from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Turbofan RUL Prediction API")

model = joblib.load("models/rf_model.pkl")


class SensorData(BaseModel):
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


def maintenance_decision(rul):
    if rul < 20:
        return "Immediate maintenance required"
    elif rul < 50:
        return "Schedule maintenance soon"
    else:
        return "Normal operation"


def add_time_series_features_single(df):
    for sensor in [f"sensor_{i}" for i in range(1, 22)]:
        df[f"{sensor}_roll_mean"] = df[sensor]
        df[f"{sensor}_roll_std"] = 0
        df[f"{sensor}_trend"] = 0

    return df


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: SensorData):
    df = pd.DataFrame([data.dict()])
    df = add_time_series_features_single(df)

    expected_features = list(model.feature_names_in_)

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    prediction = model.predict(df)[0]

    return {
        "predicted_rul": round(float(prediction), 2),
        "decision": maintenance_decision(prediction)
    }
