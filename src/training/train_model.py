from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from src.features import MODEL_FEATURES, add_time_series_features

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "CMAPSSData" / "train_FD001.txt"
MODEL_PATH = ROOT_DIR / "models" / "rf_model.pkl"
FIGURE_PATH = ROOT_DIR / "figures" / "predicted_vs_actual_RUL.png"


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    maximum_cycles = result.groupby("engine_id")["cycle"].transform("max")
    result["RUL"] = maximum_cycles - result["cycle"]
    return result


def load_data(train_file: Path = DATA_PATH) -> pd.DataFrame:
    columns = (
        ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{index}" for index in range(1, 22)]
    )
    df = pd.read_csv(train_file, sep=r"\s+", header=None, names=columns)
    return add_time_series_features(compute_rul(df))


def split_by_engine(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete engines so no engine appears in both partitions."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, validation_index = next(
        splitter.split(df, groups=df["engine_id"])
    )
    train_df = df.iloc[train_index].copy()
    validation_df = df.iloc[validation_index].copy()

    overlap = set(train_df["engine_id"]) & set(validation_df["engine_id"])
    if overlap:
        raise RuntimeError(f"Engine leakage detected: {sorted(overlap)}")
    return train_df, validation_df


def train() -> float:
    df = load_data()
    train_df, validation_df = split_by_engine(df)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[MODEL_FEATURES], train_df["RUL"])
    predictions = model.predict(validation_df[MODEL_FEATURES])
    rmse = float(
        root_mean_squared_error(validation_df["RUL"], predictions)
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        x=validation_df["RUL"],
        y=predictions,
        alpha=0.5,
    )
    maximum = max(validation_df["RUL"].max(), predictions.max())
    plt.plot([0, maximum], [0, maximum], linestyle="--")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Engine-level holdout: predicted vs actual RUL")
    plt.text(10, maximum * 0.9, f"RMSE: {rmse:.2f} cycles")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=160)
    plt.close()

    print(
        "Engine-level validation: "
        f"{validation_df['engine_id'].nunique()} held-out engines"
    )
    print(f"Validation RMSE: {rmse:.2f} cycles")
    print(f"Model saved: {MODEL_PATH}")
    return rmse


if __name__ == "__main__":
    train()
