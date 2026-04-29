import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FOLDER = "CMAPSSData"
MODEL_FOLDER = "models"
FIGURE_FOLDER = "figures"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)

MODEL_FILENAME = os.path.join(MODEL_FOLDER, "rf_model.pkl")
FIGURE_FILENAME = os.path.join(FIGURE_FOLDER, "predicted_vs_actual_RUL.png")


def compute_rul(df):
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycles, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)

    return df


def add_time_series_features(df):
    df = df.sort_values(["engine_id", "cycle"])

    for sensor in [f"sensor_{i}" for i in range(1, 22)]:
        df[f"{sensor}_roll_mean"] = (
            df.groupby("engine_id")[sensor]
            .rolling(window=5)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{sensor}_roll_std"] = (
            df.groupby("engine_id")[sensor]
            .rolling(window=5)
            .std()
            .reset_index(0, drop=True)
        )

        df[f"{sensor}_trend"] = df.groupby("engine_id")[sensor].diff()

    df = df.fillna(0)

    return df


def load_data(train_file):
    columns = (
        ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(train_file, sep=r"\s+", header=None)
    df.columns = columns

    df = compute_rul(df)
    df = add_time_series_features(df)

    return df


from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

train_file = ROOT_DIR / "data" / "raw" / "CMAPSSData" / "train_FD001.txt"
test_file = ROOT_DIR / "data" / "raw" / "CMAPSSData" / "test_FD001.txt"
rul_file = ROOT_DIR / "data" / "raw" / "CMAPSSData" / "RUL_FD001.txt"

df_train = load_data(train_file)

base_features = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

engineered_features = (
    [f"sensor_{i}_roll_mean" for i in range(1, 22)]
    + [f"sensor_{i}_roll_std" for i in range(1, 22)]
    + [f"sensor_{i}_trend" for i in range(1, 22)]
)

features = base_features + engineered_features

X = df_train[features]
y = df_train["RUL"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)

print(f"Validation RMSE: {rmse:.2f} cycles")

joblib.dump(model, MODEL_FILENAME)
print(f"Model saved: {MODEL_FILENAME}")

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.5)

max_value = max(y_val.max(), y_pred.max())
plt.plot([0, max_value], [0, max_value], linestyle="--")

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs Actual RUL")
plt.text(10, max_value * 0.9, f"RMSE: {rmse:.2f} cycles")

plt.tight_layout()
plt.savefig(FIGURE_FILENAME)
plt.savefig("predicted_vs_actual_RUL.png")
plt.show()

print(f"Figure saved: {FIGURE_FILENAME}")
print("Root figure saved: predicted_vs_actual_RUL.png")