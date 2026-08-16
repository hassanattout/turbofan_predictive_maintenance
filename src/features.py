from __future__ import annotations

import pandas as pd

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
BASE_FEATURES = SETTING_COLUMNS + SENSOR_COLUMNS
ENGINEERED_FEATURES = (
    [f"{sensor}_roll_mean" for sensor in SENSOR_COLUMNS]
    + [f"{sensor}_roll_std" for sensor in SENSOR_COLUMNS]
    + [f"{sensor}_trend" for sensor in SENSOR_COLUMNS]
)
MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
ROLLING_WINDOW = 5


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal per-engine rolling features without mixing engines."""
    required = {"engine_id", "cycle", *BASE_FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.sort_values(["engine_id", "cycle"]).copy()
    grouped = result.groupby("engine_id", sort=False)

    for sensor in SENSOR_COLUMNS:
        result[f"{sensor}_roll_mean"] = grouped[sensor].transform(
            lambda values: values.rolling(
                window=ROLLING_WINDOW,
                min_periods=1,
            ).mean()
        )
        result[f"{sensor}_roll_std"] = grouped[sensor].transform(
            lambda values: values.rolling(
                window=ROLLING_WINDOW,
                min_periods=2,
            ).std()
        )
        result[f"{sensor}_trend"] = grouped[sensor].diff()

    return result.fillna(0.0)


def latest_feature_row(history: pd.DataFrame) -> pd.DataFrame:
    """Return model features for the latest cycle in one engine history."""
    if len(history) < ROLLING_WINDOW:
        raise ValueError(
            f"At least {ROLLING_WINDOW} ordered cycles are required."
        )

    data = history.copy()
    data["engine_id"] = 1
    data["cycle"] = range(1, len(data) + 1)
    engineered = add_time_series_features(data)
    return engineered.iloc[[-1]][MODEL_FEATURES]


def maintenance_decision(rul: float) -> str:
    if rul < 20:
        return "Immediate maintenance required"
    if rul < 50:
        return "Schedule maintenance soon"
    return "Normal operation"
